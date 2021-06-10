#!/bin/bash

# The base directory.
# Don't include a trailing '/'.
BASE_DIR="."

# Whether to enable (0) or disable (non-0) debug logging.
DEBUG_LOGGING=1

BASE_DIR=$(readlink -f "$BASE_DIR") && \
echo "Making sure the base data is available..." && \
bash extract_dataset.sh "$BASE_DIR" $DEBUG_LOGGING && \
echo "Done! Going to split data now..." && \
bash split_data.sh "$BASE_DIR" $DEBUG_LOGGING && \
echo "Done! Going to deduplicate files now..." && \
bash deduplicate_files.sh "$BASE_DIR" $DEBUG_LOGGING && \
echo "Done! Going to preprocess the images now..." && \
python preprocess_images.py "$BASE_DIR/data/dataset" $DEBUG_LOGGING && \
echo "Done! Going to apply ImageMorph data augmentation now..." && \
bash apply_data_augmentation.sh "$BASE_DIR" $DEBUG_LOGGING && \
echo "Done! Going to apply the other data augmentation methods now..." && \
python data_augmentation.py "$BASE_DIR/data/dataset" $DEBUG_LOGGING && \
echo "Done!"
