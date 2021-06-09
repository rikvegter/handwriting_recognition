#!/bin/bash

# The base directory.
# Don't include a trailing '/'.
DATA_DIR="/home/pim/Documents/workspace/handwriting_recognition"

# Whether to enable (0) or disable (non-0) debug logging.
DEBUG_LOGGING=1

echo "Making sure the base data is available..." && \
bash extract_dataset.sh "$DATA_DIR" $DEBUG_LOGGING && \
echo "Done! Going to split data now..." && \
bash split_data.sh "$DATA_DIR" $DEBUG_LOGGING && \
echo "Done! Going to deduplicate files now..." && \
bash deduplicate_files.sh "$DATA_DIR" $DEBUG_LOGGING && \
echo "Done! Going to preprocess the images now..." && \
python preprocess_images.py "$DATA_DIR" $DEBUG_LOGGING && \
echo "Done! Going to apply ImageMorph data augmentation now..." && \
bash apply_data_augmentation.sh "$DATA_DIR" $DEBUG_LOGGING && \
echo "Done! Going to apply the other data augmentation methods now..." && \
python data_augmentation.py "$DATA_DIR" $DEBUG_LOGGING && \
echo "Done!"
