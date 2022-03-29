# CoopFlow_paddle
Cooperative learning of Langevin Flow (short-run MCMC) and Normalizing Flow

This repository contains a paddle-paddle implementation for the CoopFlow algorithm proposed in the ICLR 2022 paper "[A Tale of Two Flows: Cooperative Learning of Langevin Flow and Normalizing Flow Toward Energy-Based Model](https://openreview.net/forum?id=31d5RLCUuXC&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2022%2FConference%2FAuthors%23your-submissions))"

## Set Up Environment
Please install the following packages:

python = 3.6
paddlepaddle-gpu = 2.2.2
matplotlib = 3.2.2

For the installation of paddle, please check "[this website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/windows-conda.html)".

## train to model
1. To train the CoopFlow model on cifar10:
```bash
python main.py --train True --resume False
```
2. To train the CoopFlow(pre) model on cifar10:
```bash
python main.py --train True --resume False --load_pretrain_flow True
```
Note that to train CoopFlow model, you should first have a pretrained flow model. We provide the pretrained checkpoint here (coming soon). Or you can train the checkpoint you self using the code main_flow.py. Simply run the following code. 
```bash
python main_flow.py
```
It will create ./pretrain_flow folder and save the checkpoint under directory ./pretrain_flow/save_flow. Please copy the generated checkpoint into ./flow_ckpt directory so that the CoopFlow code can find it.

## Sample image generation results
**Cifar-10 CoopFlow** (Left: initial proposal generated by normalizing flow; Right: modified examples by Langevin flow) 

<img src="/images/exp_cifar/flow.png" width="300"/> <img src="/images/exp_cifar/ebm.png" width="300"/>

**Cifar-10 CoopFlow(pre)** (Left: initial proposal generated by normalizing flow; Right: modified examples by Langevin flow) 

<img src="/images/exp_cifar_load/flow.png" width="300"/> <img src="/images/exp_cifar_load/ebm.png" width="300"/>
