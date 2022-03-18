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

    testset = paddle.vision.datasets.Cifar10(mode='test', download=True, transform=transform_test)
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

    loss_fn = util.NLLLoss()
    
    warm_up = args.warm_up * args.batch_size
    def decay_fn(step):
        return min(1., step / warm_up)
    flow_scheduler = paddle.optimizer.lr.LambdaDecay(args.lr_flow, lr_lambda=decay_fn)
    if args.max_grad_norm > 0:
        norm_clipper = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.max_grad_norm)
    else:
        norm_clipper = None
    # rememeber to deal with the flow_optimizer here later(split into group; weight decay; gradient clip)
    flow_param_groups, flow_name_register = util.get_param_groups(flow_net, args.weight_decay, norm_suffix='weight_g')
    flow_optimizer = optimizer.AdamW(parameters=flow_net.parameters(), learning_rate=flow_scheduler, grad_clip = norm_clipper, apply_decay_param_fun=flow_name_register.name_check, weight_decay=args.weight_decay)

    start_epoch = 0
    gnorms_flow = []
    if args.resume:
        # Load checkpoint.
        file_name = "./pretrain_flow/save_flow/84_1785.pdparams"
        print('Resuming from checkpoint at {}...'.format(file_name))
        checkpoint = util.load(file_name)
        flow_net.set_state_dict(checkpoint['flow_net'])
        flow_optimizer.set_state_dict(checkpoint['flow_optimizer'])
        global gstep
        start_epoch = checkpoint['epoch'] + 1
        gstep = start_epoch * len(trainset)

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        gnorms_flow = train(epoch, flow_net, trainloader, flow_optimizer, flow_scheduler,
              loss_fn, args, save_checkpoint=(epoch % 2 == 0), gnorms_flow=gnorms_flow)


def train(epoch, flow_net, trainloader, flow_optimizer, flow_scheduler, loss_fn, args, save_checkpoint=True, gnorms_flow=None):
    global gstep
    print('\nEpoch: %d' % epoch)
    flow_net.train()
    flow_loss_meter = util.AverageMeter()
    counter = 0
    start_time = time.time()
    num_iter = math.ceil(float(len(trainloader.dataset)) / args.batch_size)
    tmp_xs = []
    for x, _ in trainloader:
        # train flow model
        if epoch == 0 and counter < 50:
            if counter == 0:
                print('Data dependent initialization for flow parameter at the begining of training')
            tmp_xs.append(x.detach().clone())
            counter += 1
            if counter == 50:
                tmp_xs = paddle.concat(tmp_xs, axis=0)
                with paddle.no_grad():
                    flow_net(tmp_xs.detach(), reverse=False)
                print('Begin training')
            continue

        flow_optimizer.clear_grad()
        z, sldj = flow_net(x.detach(), reverse=False)
        flow_loss = loss_fn(z, sldj)
        flow_loss.backward()

        flow_grad_norm = 0.0
        for p in flow_net.parameters():
            flow_grad_norm = flow_grad_norm + (paddle.linalg.norm(p.grad.clone().detach()).item()) ** 2
        flow_grad_norm = np.sqrt(flow_grad_norm)

        if flow_grad_norm < 3e5: 
            flow_optimizer.step()
            flow_scheduler.step(epoch=gstep)

        if counter % 200 == 0:
            x_flow = sample_flow(flow_net, args.batch_size).detach()
            x_array = x_flow.clone().detach().flatten().numpy()
            os.makedirs(args.save_dir, exist_ok=True)
            plt.hist(np.clip(x_array, -2, 2), bins=128)
            plt.savefig('{}/hist_flow.png'.format(args.save_dir))
            plt.close()
            plt.plot(np.arange(len(gnorms_flow)), gnorms_flow)
            plt.savefig('{}/grad_norm_flow.png'.format(args.save_dir))
            plt.close()

            #print(gnorms_flow, gnorms_ebm)
            np.save('{}/grad_norm_flow.npy'.format(args.save_dir), np.array(gnorms_flow))
            img_grid = util.make_grid(0.5 * (paddle.clip(x_flow, -1, 1.) + 1.0), nrow = int(args.batch_size ** 0.5), padding=2, pad_value=255, normalize=True)
            util.save_image(img_grid, '{}/flow_epoch_{}.png'.format(args.save_dir, epoch))
            img_grid = util.make_grid(0.5 * (paddle.clip(x, -1, 1.) + 1.0), nrow = int(args.batch_size ** 0.5), padding=2, pad_value=255, normalize=True)
            util.save_image(img_grid, '{}/ori_epoch_{}.png'.format(args.save_dir, epoch))

        if counter % 20 == 0:
            print('Epoch {} iter {}/{} time{:.3f} FLOW: flow_loss {:.3f}' \
                .format(epoch, counter, num_iter, time.time() - start_time, flow_loss.item()))

        if save_checkpoint and counter == num_iter - 1:
            print('Saving...')
            state = {
                'flow_net': flow_net.state_dict(),
                'flow_optimizer': flow_optimizer.state_dict(),
                'epoch': epoch,
            }
            os.makedirs(args.ckpt_dir, exist_ok=True)
            util.save(state, '{}/{}_{}.pdparams'.format(args.ckpt_dir, epoch, counter))

        gstep += x.shape[0]
        counter += 1
    return gnorms_flow


@paddle.no_grad()
def sample_flow(net, batch_size):
    """
    Sample from Flow model.
    """
    z = paddle.randn(shape=[batch_size, 3, 32, 32], dtype='float32')
    x, _ = net(z, reverse=True)
    x = 2.0 * F.sigmoid(x) - 1.0
    return x


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
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='pretrain_flow/flow_samples', help='Directory for saving samples')
    parser.add_argument('--ckpt_dir', type=str, default='pretrain_flow/save_flow', help='Directory for saving samples')
    parser.add_argument('--use_attn', type=str2bool, default=True, help='Use attention in the coupling layers')
    parser.add_argument('--warm_up', type=int, default=200, help='Number of batches for LR warmup')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')
    parser.add_argument('--num_steps_Langevin_coopNet', default=30, type=int,
                        help='number of Langevin steps in CoopNets')
    parser.add_argument('--step_size', default=0.03, type=float, help='Langevin step size')
    parser.add_argument('--train', default=True, type=bool, help='whether to train the model or to generate image for computing fid')

    best_loss = 0
    gstep = 0

    main(parser.parse_args())