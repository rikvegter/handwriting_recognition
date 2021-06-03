#!/bin/bash

#bash split_data.sh && bash find_duplicate_leaks.sh && bash apply_data_augmentation.sh
bash split_data.sh && \
bash deduplicate_files.sh && \
python preprocess_images.py && \
python data_augmentation.py
