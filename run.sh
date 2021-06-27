#!/bin/bash

function download_image() {
    echo "Image not found... Going to download it now, this may take a second!"
    image_archive="save_tree_save_world_hwr.docker"
    wget -q --show-progress -O "$image_archive" http://dl.pim16aap2.nl/s/SKmdkX4t7bi3G9Z/download/save_tree_save_world_hwr.docker
    test $(wc -c "$image_archive" | awk '{print $1}') -gt 1000000 || { echo "Failed to download file! Please get it manually here: "; echo "https://drive.google.com/file/d/1ckqogv-WzmlrzskR_wF4V4e5vtIH5Oh2/view?usp=sharing"; echo "And then import it this way 'docker load < $image_archive'"; exit 1; }
    echo "The image has been downloaded! Going to import it now..."
    docker load < "$image_archive"
    rm "$image_archive"
}

# Ensure an output directory is specified.
test -z $1 && { echo "No input directory specified! Usage: \"./run.sh [/]path/to/images/\""; exit 2; }

# Make sure the container doesn't already exist (as we're deleting it afterwards).
docker container inspect save_tree_save_world_hwr > /dev/null 2>&1 && { echo "You already have a container named \"save_tree_save_world_hwr\"! Aborting!"; exit 3; }

# Download the image if the user does not have it yet.
docker images | grep save_tree_save_world_hwr || download_image

# Run the pipeline.
docker run --user $(id -u):$(id -g) --name save_tree_save_world_hwr -v "$(pwd)/results/":/output -v "$1":/input save_tree_save_world_hwr

# Clean up the docker container.
docker rm save_tree_save_world_hwr
