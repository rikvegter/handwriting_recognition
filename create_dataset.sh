#!/bin/bash

echo "Going to split data now..." && \
bash split_data.sh && \
echo "Done! Going to deduplicate files now..." && \
bash deduplicate_files.sh && \
echo "Done! Going to preprocess the images now..." && \
python preprocess_images.py && \
echo "Done! Going to apply ImageMorph data augmentation now..." && \
bash apply_data_augmentation.sh && \
echo "Done! Going to apply the other data augmentation methods now..." && \
python data_augmentation.py && \
echo "Done!"
