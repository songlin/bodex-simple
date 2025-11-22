#!/bin/bash

if [ "$1" = "inspire" ]; then
    CUDA_VISIBLE_DEVICES=8 python example_grasp/plan_batch_env.py \
        --manip_cfg_file sim_inspire/left.yml \
        --parallel_world 1 \
        --skip \
        --save_id 0

elif [ "$1" = "dex3" ]; then
    CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py \
        --manip_cfg_file sim_dex3/right.yml \
        --parallel_world 1 \
        --skip \
        --save_id 0

elif [ "$1" = "dexmate" ]; then
    CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py \
        --manip_cfg_file sim_dexmate/left.yml \
        --parallel_world 1 \
        --skip \
        --save_id 0

else
    echo "Usage: $0 {inspire|dex3|dexmate}"
    exit 1
fi