#!/bin/bash
n_gpu=2  # number of gpu to use
#python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_ycb.py --gpus=$n_gpu --comment $1
python3 -m torch.distributed.launch --nproc_per_node=1 train_ycb.py --gpus=1 --comment $1 -test_pose
