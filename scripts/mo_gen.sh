#!/bin/bash

if [ "$1" = "inspire" ]; then
    rm src/curobo/content/assets/output/sim_inspire/left/inspire_debug/graspdata/apple_*

    CUDA_VISIBLE_DEVICES=7 python example_grasp/plan_mogen_batch.py \
        --manip_cfg_file sim_inspire/left.yml \
        --task grasp_and_mogen \
        --skip

elif [ "$1" = "dex3" ]; then
    rm src/curobo/content/assets/output/sim_dex3/right/dex3_debug/graspdata/apple_*

    CUDA_VISIBLE_DEVICES=6 python example_grasp/plan_mogen_batch.py \
        --manip_cfg_file sim_dex3/right.yml \
        --task grasp_and_mogen \
        --skip

elif [ "$1" = "dexmate" ]; then
    rm src/curobo/content/assets/output/sim_vega1/right/dexmate_debug/graspdata/apple_*

    CUDA_VISIBLE_DEVICES=5 python example_grasp/plan_mogen_batch.py \
        --manip_cfg_file sim_vega1/right.yml \
        --task grasp_and_mogen \
        --skip

else
    echo "Usage: $0 {inspire|dex3|dexmate}"
    exit 1
fi