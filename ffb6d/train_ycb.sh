#!/bin/bash
n_gpu=2  # number of gpu to use
#python3 train_ycb.py --gpus=$n_gpu 
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_ycb.py --gpus=$n_gpu --comment $1 -test_pose
