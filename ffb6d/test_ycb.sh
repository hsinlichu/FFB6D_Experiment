#!/bin/bash
#tst_mdl=FFB6D_pretrained_best.pth.tar  # checkpoint to test.
tst_mdl=/home/openark/Desktop/jameschu/FFB6D/ffb6d/train_log/ycb/log/0307_190053_test/FFB6D_best_6_106872_38.0164.pth.tar  # checkpoint to test.
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=8888 train_ycb.py --gpu '1' -eval_net -checkpoint $tst_mdl -test -test_pose --comment $1 # -debug
