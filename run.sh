#!/bin/bash

cmd="python3 main.py $@"
docker run --cap-add SYS_ADMIN --device /dev/fuse --security-opt apparmor:unconfined -v $(pwd):/files save_tree_save_world_hwr bash -c "bash /mergerfs.sh && cd /hwr_run && $cmd"
