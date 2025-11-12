#!/bin/bash

if [ "$1" = "inspire" ]; then
    python example_grasp/visualize_npy.py \
        --manip_cfg_file sim_inspire/left.yml \
        --path inspire_debug \
        --mode mogen \
        --skip

elif [ "$1" = "dex3" ]; then
    python example_grasp/visualize_npy.py \
        --manip_cfg_file sim_dex3/right.yml \
        --path dex3_debug \
        --mode mogen \
        --skip

else
    echo "Usage: $0 {inspire|dex3}"
    exit 1
fi