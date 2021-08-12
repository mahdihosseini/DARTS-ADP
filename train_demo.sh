#!/bin/bash

python train.py \
--data ./data \
--dataset ADP \
--arch DARTS_ADP_N4 \
--batch_size 96 --epochs 600 \
--learning_rate 0.025 \
--cutout \
--seed 0 \
--save train_demo