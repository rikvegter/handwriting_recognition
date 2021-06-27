#!/bin/bash

docker container inspect save_tree_save_world_hwr > /dev/null 2>&1 && { echo "You already have a container named \"save_tree_save_world_hwr\"! Aborting!"; exit 1; }

cmd="python3 main.py $@"

docker run --cap-add SYS_ADMIN --name save_tree_save_world_hwr --device /dev/fuse --security-opt apparmor:unconfined -v $(pwd):/files save_tree_save_world_hwr bash -c "bash /mergerfs.sh && cd /hwr_run && $cmd"

docker rm save_tree_save_world_hwr
