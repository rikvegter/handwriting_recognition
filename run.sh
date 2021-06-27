#!/bin/bash

# Ensure an output directory is specified.
test -z $1 && { echo "No input directory specified! Usage: \"./run.sh [/]path/to/images/\""; exit 1; }

# Make sure the container doesn't already exist (as we're deleting it afterwards).
docker container inspect save_tree_save_world_hwr > /dev/null 2>&1 && { echo "You already have a container named \"save_tree_save_world_hwr\"! Aborting!"; exit 2; }

# Create the results dir.
test -d results || { mkdir results || exit 3; }

# Run the pipeline.
docker run --name save_tree_save_world_hwr -v "$(pwd)/results/":/output -v "$1":/input save_tree_save_world_hwr

# Clean up the docker container.
docker rm save_tree_save_world_hwr
