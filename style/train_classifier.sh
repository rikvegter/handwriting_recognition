#!/bin/bash
python3 scripts/preprocess_images.py -ls -n style_fragments images/style_labeled_graphemes ./data/codebook
python3 scripts/extract_fraglets.py ./data/codebook -a 5


python3 scripts/preprocess_images.py -s -n style_fragments images/style_fragments ./data/style-fragments
python3 scripts/extract_fraglets.py ./data/style-fragments -a 5

# Train the classifier
python3 train_classifier.py ./data/style-fragments ./data/codebook ./classifier -p 4.5 -a 500 -n 10 -c 2 -v 50 -k 300

# To view preprocessed images, extracted fraglets, or segementation, run the following:
#python3 scripts/plot_data.py ./data/style-fragments images
#python3 scripts/plot_data.py ./data/style-fragments segmentation
#python3 scripts/plot_data.py ./data/style-fragments fraglets

#python3 scripts/plot_data.py ./data/codebook images
#python3 scripts/plot_data.py ./data/codebook segmentation
#python3 scripts/plot_data.py ./data/codebook fraglets