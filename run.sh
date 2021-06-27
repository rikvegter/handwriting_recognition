#!/bin/bash

docker container inspect save_tree_save_world_hwr > /dev/null 2>&1 && { echo "You already have a container named \"save_tree_save_world_hwr\"! Aborting!"; exit 1; }

docker run --name save_tree_save_world_hwr -v "$(pwd)/results/":/output -v "$1":/input save_tree_save_world_hwr

docker rm save_tree_save_world_hwr
