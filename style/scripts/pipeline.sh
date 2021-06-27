#!/bin/bash

## Preprocessing

# Unlabeled fragments
#python3 preprocess_images.py  ~/hwr/image-data ../data/fragments -n fragments -o 0 -c 0 -s 0.5 -p 
# Style-labeled allographs
python3 preprocess_images.py -ls ./characters-for-style-classification ../data/codebook -n codebook -o 0 -c 0 -g 0.5
#Style-labeled fragments
python3 preprocess_images.py -s ~/hwr/full_periods ../data/style-fragments -n style-fragments -o 0 -c 0 -g 0.5

