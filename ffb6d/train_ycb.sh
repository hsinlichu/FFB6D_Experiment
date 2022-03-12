#!/bin/bash
n_gpu=1  # number of gpu to use
#python3 train_ycb.py --gpus=$n_gpu 
python3 -m torch.distributed.launch --nproc_per_node=1 train_ycb.py --gpus=1 --comment $1 -test_pose
