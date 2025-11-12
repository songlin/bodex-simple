#!/bin/bash

if [ "$1" = "inspire" ]; then
    CUDA_VISIBLE_DEVICES=6 python example_grasp/plan_mogen_batch.py \
        --manip_cfg_file sim_inspire/left.yml \
        --task grasp_and_mogen \
        --skip

elif [ "$1" = "dex3" ]; then
    CUDA_VISIBLE_DEVICES=6 python example_grasp/plan_mogen_batch.py \
        --manip_cfg_file sim_dex3/right.yml \
        --task grasp_and_mogen \
        --skip

else
    echo "Usage: $0 {inspire|dex3}"
    exit 1
fi