#!/bin/bash

python test.py \
--data ./data \
--dataset ADP \
--arch DARTS_ADP_N4 \
--model_path ./pretrained/ADP/darts_adp_n4.pt

