"""
Train CoopFlow on CIFAR-10.
"""
import argparse
import numpy as np
import os
import time
import random
import paddle
import paddle.optimizer as optimizer
import paddle.vision.transforms as transforms
import paddle.nn.functional as F 
import util
import math
from models import FlowPlusPlus
from models import EBM as EBM
from matplotlib import pyplot as plt


def main(args):

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]) 

    trainset = paddle.vision.datasets.Cifar10(mode='train', download=True, transform=transform_train)
    trainloader = paddle.io.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = paddle.vision.datasets.Cifar10(mode='train', download=True, transform=transform_test)
    testloader = paddle.io.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) # testing data do not shuffle or apply random transpose

    # Model
    print('Building model..')
    flow_net = FlowPlusPlus(scales=[(0, 4), (2, 3)],
                       in_shape=(3, 32, 32),
                       mid_channels=args.num_channels,
                       num_blocks=args.num_blocks,
                       num_dequant_blocks=args.num_dequant_blocks,
                       num_components=args.num_components,
                       use_attn=args.use_attn,
                       drop_prob=args.drop_prob)
    ebm_net = EBM(n_c=3, n_f=128)

    loss_fn = util.NLLLoss()
    
    warm_up = args.warm_up * args.batch_size
    def decay_fn(step):
        return min(1., step / warm_up)
    flow_scheduler = paddle.optimizer.lr.LambdaDecay(args.lr_flow, lr_lambda=decay_fn)
    ebm_scheduler = paddle.optimizer.lr.LambdaDecay(args.lr_ebm, lr_lambda=decay_fn)
    if args.max_grad_norm > 0:
        norm_clipper = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.max_grad_norm)
    else:
        norm_clipper = None
    flow_param_groups, flow_name_register = util.get_param_groups(flow_net, args.weight_decay, norm_suffix='weight_g')
    flow_optimizer = optimizer.AdamW(parameters=flow_net.parameters(), learning_rate=flow_scheduler, grad_clip = norm_clipper, apply_decay_param_fun=flow_name_register.name_check, weight_decay=args.weight_decay)
    ebm_optimizer = optimizer.Adam(parameters=ebm_net.parameters(), learning_rate=ebm_scheduler, beta1=0.5, beta2=0.5)

    start_epoch = 0
    best_fid = np.inf
    fids = []
    gnorms_flow = []
    gnorms_ebm = []
    if args.load_pretrain_flow:
        file_name = "./flow_ckpt/84_1785.pdparams"
        print('Resuming from checkpoint at {}...'.format(file_name))
        checkpoint = util.load(file_name)
        flow_net.set_state_dict(checkpoint['flow_net'])
    
    if args.resume:
        # Load checkpoint.
        file_name = "./exp_cifar_load/save/118_1785.pdparams"
        print('Resuming from checkpoint at {}...'.format(file_name))
        checkpoint = util.load(file_name)
        flow_net.set_state_dict(checkpoint['flow_net'])
        ebm_net.set_state_dict(checkpoint['ebm_net'])
        flow_optimizer.set_state_dict(checkpoint['flow_optimizer'])
        ebm_optimizer.set_state_dict(checkpoint['ebm_optimizer']) 
        global gstep
        start_epoch = checkpoint['epoch'] + 1
        gstep = start_epoch * len(trainset)

    if not args.train:
        assert args.resume
        test(ebm_net, flow_net, testloader, args)
        return

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        gnorms_flow, gnorms_ebm = train(epoch, ebm_net, flow_net, trainloader, ebm_optimizer, ebm_scheduler, flow_optimizer, flow_scheduler,
              loss_fn, args, save_checkpoint=(epoch % 2 == 0), gnorms_flow=gnorms_flow, gnorms_ebm=gnorms_ebm)

def train(epoch, ebm_net, flow_net, trainloader, ebm_optimizer, ebm_scheduler, flow_optimizer, flow_scheduler, loss_fn, args, save_checkpoint=True, gnorms_flow=None, gnorms_ebm=None):
    global gstep
    print('\nEpoch: %d' % epoch)
    ebm_net.train()
    flow_net.train()
    flow_loss_meter = util.AverageMeter()
    ebm_loss_meter = util.AverageMeter()
    counter = 0
    start_time = time.time()
    num_iter = math.ceil(float(len(trainloader.dataset)) / args.batch_size)
    tmp_xs = []
    for x, _ in trainloader:
        # train flow model
        if epoch == 0 and counter < 20:
            if counter == 0:
                print('Data dependent initialization for flow parameter at the begining of training')
            tmp_xs.append(x.detach().clone())
            counter += 1
            if counter == 20:
                tmp_xs = paddle.concat(tmp_xs, axis=0)
                with paddle.no_grad():
                    flow_net(tmp_xs.detach(), reverse=False)
                print('Begin training')
            continue

        x_flow = sample_flow(flow_net, args.batch_size).detach()
        save_img = (counter % 200 == 0)
        x_ebm = ebm_sample(epoch, ebm_net, K=args.num_steps_Langevin_coopNet, step_size=args.step_size, p_0=x_flow\
                           , num_sample=args.batch_size)
        
        mse_loss = paddle.sum(paddle.mean((x_ebm.detach() - x_flow) ** 2, axis=0)).detach() # just monitor image change, not influence optimization
        flow_optimizer.clear_grad()
        z, sldj = flow_net(x_ebm.detach(), reverse=False)
        flow_loss = loss_fn(z, sldj)
        flow_loss_meter.update(flow_loss.item(), x_ebm.shape[0])
        flow_loss.backward()

        flow_grad_norm = 0.0
        for p in flow_net.parameters():
            flow_grad_norm = flow_grad_norm + (paddle.linalg.norm(p.grad.clone().detach()).item()) ** 2
        flow_grad_norm = np.sqrt(flow_grad_norm)

        # train ebm model
        ebm_optimizer.clear_grad()
        en_pos = ebm_net(x.detach()).mean()
        en_neg = ebm_net(x_ebm.detach()).mean()
        ebm_loss = en_neg - en_pos
        ebm_loss_meter.update(ebm_loss.item(), x_ebm.shape[0])
        ebm_loss.backward()
         
        ebm_grad_norm = 0.0
        for p in ebm_net.parameters():
            ebm_grad_norm = ebm_grad_norm + (paddle.linalg.norm(p.grad.clone().detach()).item()) ** 2
        ebm_grad_norm = np.sqrt(ebm_grad_norm)   

        gnorms_ebm.append(ebm_grad_norm)
        gnorms_flow.append(flow_grad_norm)
        if ebm_grad_norm < 5e6:
            ebm_optimizer.step()
            ebm_scheduler.step(epoch=gstep)
            if (epoch > 10 or not args.load_pretrain_flow) and flow_grad_norm < 5e6: 
                flow_optimizer.step()
                flow_scheduler.step(epoch=gstep)

        if paddle.isnan(mse_loss) or paddle.isnan(flow_loss) or paddle.isnan(en_pos) or paddle.isnan(en_neg) or paddle.isnan(ebm_loss):
            x_array = x_flow.clone().detach().flatten().numpy()
            os.makedirs(args.save_dir, exist_ok=True)
            plt.hist(np.clip(x_array, -2, 2), bins=128)
            plt.savefig('{}/hist_flow_before_blow.png'.format(args.save_dir))
            plt.close()
            x_ebm_array = x_ebm.clone().detach().flatten().numpy()
            plt.figure()
            plt.hist(np.clip(x_ebm_array, -2, 2), bins=128)
            plt.savefig('{}/hist_ebm_before_blow.png'.format(args.save_dir))
            plt.close()
            assert False

        if counter % 200 == 0:
            x_array = x_flow.clone().detach().flatten().numpy()
            os.makedirs(args.save_dir, exist_ok=True)
            plt.hist(np.clip(x_array, -2, 2), bins=128)
            plt.savefig('{}/hist_flow.png'.format(args.save_dir))
            plt.close()
            x_ebm_array = x_ebm.clone().detach().flatten().numpy()
            plt.figure()
            plt.hist(np.clip(x_ebm_array, -2, 2), bins=128)
            plt.savefig('{}/hist_ebm.png'.format(args.save_dir))
            plt.close()
            x_array = x.clone().detach().flatten().numpy()
            plt.figure()
            plt.hist(np.clip(x_array, -2, 2), bins=128)
            plt.savefig('{}/hist_ori.png'.format(args.save_dir))
            plt.close()
            plt.figure()
            plt.plot(np.arange(len(gnorms_flow)), gnorms_flow)
            plt.savefig('{}/grad_norm_flow.png'.format(args.save_dir))
            plt.close()

            plt.figure()
            plt.plot(np.arange(len(gnorms_ebm)), gnorms_ebm)
            plt.savefig('{}/grad_norm_ebm.png'.format(args.save_dir))
            plt.close()
            np.save('{}/grad_norm_flow.npy'.format(args.save_dir), np.array(gnorms_flow))
            np.save('{}/grad_norm_ebm.npy'.format(args.save_dir), np.array(gnorms_ebm))
            img_grid = util.make_grid(0.5 * (paddle.clip(x_ebm, -1, 1.) + 1.0), nrow = int(args.batch_size ** 0.5), padding=2, pad_value=255, normalize=True)
            util.save_image(img_grid, '{}/ebm_epoch_{}.png'.format(args.save_dir, epoch))
            img_grid = util.make_grid(0.5 * (paddle.clip(x_flow, -1, 1.) + 1.0), nrow = int(args.batch_size ** 0.5), padding=2, pad_value=255, normalize=True)
            util.save_image(img_grid, '{}/flow_epoch_{}.png'.format(args.save_dir, epoch))
            img_grid = util.make_grid(0.5 * (paddle.clip(x, -1, 1.) + 1.0), nrow = int(args.batch_size ** 0.5), padding=2, pad_value=255, normalize=True)
            util.save_image(img_grid, '{}/ori_epoch_{}.png'.format(args.save_dir, epoch))

        if counter % 20 == 0:
            print('Epoch {} iter {}/{} time{:.3f} FLOW: image mse change {:.3f} flow_loss {:.3f} EBM: pos en {:.3f} neg en{:.3f} en diff {:.3f}' \
                .format(epoch, counter, num_iter, time.time() - start_time, mse_loss.item(), flow_loss.item(), en_pos.item(), en_neg.item(), ebm_loss.item()))

        if save_checkpoint and counter == num_iter - 1:
            print('Saving...')
            state = {
                'ebm_net': ebm_net.state_dict(),
                'flow_net': flow_net.state_dict(),
                'ebm_optimizer': ebm_optimizer.state_dict(),
                'flow_optimizer': flow_optimizer.state_dict(),
                'epoch': epoch,
            }
            os.makedirs(args.ckpt_dir, exist_ok=True)
            util.save(state, '{}/{}_{}.pdparams'.format(args.ckpt_dir, epoch, counter))

        gstep += x.shape[0]
        counter += 1
    return gnorms_flow, gnorms_ebm


def ebm_sample(epoch, net, K=10, step_size=0.02, p_0=None, num_sample=100):
    x_k = p_0.detach().clone()
    for k in range(K):
        x_k.stop_gradient = False
        grad = paddle.grad([net(x_k).sum()], [x_k])[0]
        delta = -0.5 * step_size * step_size * grad 
        delta = paddle.clip(delta, -0.1, 0.1)
        x_k = (x_k - delta).detach()
        x_k = paddle.clip(x_k, -1.0, 1.0)
    return x_k.detach()


@paddle.no_grad()
def sample_flow(net, batch_size):
    """
    Sample from Flow model.
    """
    z = paddle.randn(shape=[batch_size, 3, 32, 32], dtype='float32')
    x, _ = net(z, reverse=True)
    x = 2.0 * F.sigmoid(x) - 1.0
    return x


def test_in_code(ebm_net, flow_net, args, real_m, real_s):
    flow_net.eval()
    ebm_net.eval()
    gen_img = []
    start_time = time.time()
    print("Begin testing")
    for i in range(2):
        batch_size = 250
        with paddle.no_grad():
            x_flow = sample_flow(flow_net, batch_size).detach()
        x_ebm = ebm_sample(0, ebm_net, K=args.num_steps_Langevin_coopNet, step_size=0.133, p_0=x_flow, num_sample=batch_size)
        x_ebm = (x_ebm + 1.0) * 0.5
        gen_img.append(x_ebm.detach().clone())
        if i % 1 == 0 or i == 199:
            print(i * batch_size, time.time() - start_time)
    gen_img = paddle.concat(gen_img, axis=0)
    gen_img = torch.tensor(gen_img.detach().numpy())

    fid = pfw.fid(gen_img, real_m = real_m, real_s=real_s, device="cuda:0")
    flow_net.train()
    ebm_net.train()
    return fid


def test(ebm_net, flow_net, testloader, args):
    flow_net.eval()
    ebm_net.eval()
    ori_dir = './exp_cifar_load/ori_samples'
    gen_dir = './exp_cifar_load/gen_samples'
    os.makedirs(ori_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    total_imgs = 50000
    counter = 0
    grid_img = []
    grid_img_flow = []
    start_time = time.time()
    for x, _ in testloader:
        batch_size = len(x)
        with paddle.no_grad():
            x_flow = sample_flow(flow_net,batch_size).detach()
        x_ebm = ebm_sample(0, ebm_net, K=args.num_steps_Langevin_coopNet, step_size=args.step_size, p_0=x_flow, num_sample=batch_size)
        for i in range(batch_size):
            util.save_image((x[i] + 1.0) * 0.5, os.path.join(ori_dir, '{}.png'.format(counter)))
            util.save_image((x_ebm[i] + 1.0) * 0.5, os.path.join(gen_dir, '{}.png'.format(counter)))

            if len(grid_img) < 100:
                grid_img.append((x_ebm[i] + 1.0) * 0.5)
                grid_img_flow.append((x_flow[i] + 1.0) * 0.5)
            counter += 1
            if counter % 1000 == 0:
                print(counter, time.time() - start_time)
            if counter >= total_imgs:
                img_grid = util.make_grid(grid_img, nrow = int(len(grid_img) ** 0.5), padding=2, pad_value=255, normalize=True)
                util.save_image(img_grid, './exp_cifar_load/ebm.png')
                img_grid = util.make_grid(grid_img_flow, nrow = int(len(grid_img) ** 0.5), padding=2, pad_value=255, normalize=True)
                util.save_image(img_grid, './exp_cifar_load/flow.png')
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoopFlow on cifar10')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=28, type=int, help='Batch size per GPU')
    parser.add_argument('--image_size', default=32, type=int, help='Image size')
    parser.add_argument('--lr_flow', default=1e-4, type=float, help='Peak learning rate')
    parser.add_argument('--lr_ebm', default=1e-4, type=float, help='Peak learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1., help='Max gradient norm for clipping')
    parser.add_argument('--drop_prob', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--num_blocks', default=10, type=int, help='Number of blocks in Flow++')
    parser.add_argument('--num_components', default=32, type=int, help='Number of components in the mixture')
    parser.add_argument('--num_dequant_blocks', default=2, type=int, help='Number of blocks in dequantization')
    parser.add_argument('--num_channels', default=96, type=int, help='Number of channels in Flow++')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=True, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='exp_cifar/Coop_samples', help='Directory for saving samples')
    parser.add_argument('--ckpt_dir', type=str, default='exp_cifar/save', help='Directory for saving samples')
    parser.add_argument('--use_attn', type=str2bool, default=True, help='Use attention in the coupling layers')
    parser.add_argument('--warm_up', type=int, default=200, help='Number of batches for LR warmup')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')
    parser.add_argument('--num_steps_Langevin_coopNet', default=30, type=int,
                        help='number of Langevin steps in CoopNets')
    parser.add_argument('--step_size', default=0.1, type=float, help='Langevin step size')
    parser.add_argument('--load_pretrain_flow', default=False, type=bool, help='Using CoopFlow (set to false) or CoopFlow-pre settings (set to true)')
    parser.add_argument('--train', default=False, type=bool, help='whether to train the model or to generate image for computing fid')

    best_loss = 0
    gstep = 0

    main(parser.parse_args())