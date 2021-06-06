# Problem specification

Goal is to go from a fragment to a label (Archaic, Hasmodean, Herodian), or to a probability distribution over the labels. 

This classification is done on the basis of the example graphemes. I don't know what features will work well in our case so should compare different kinds of features. We don't have a lot of features, so some kind of unsupervised learning might be useful. 

The dataset is also very unbalanced, with some letters missing entirely in the "Archaic" style. The numbers of grapheme examples (cut out and labeled with character) are as follows :

 - Archaic: 348
 - Hasmodean: 1.371
 - Herodian: 1.377
 - Unlabeled style: 5.564 (I don't know if these are all distinct from the labeled graphemes)
 
Fragments: 20 fragments with roughly on the order of 100-200 graphemes each, so the number of graphemes in the fragments is of the same order of magnitude as those that have been cut out. 

# Types of features
## Textural features
Textural featues like the Sobel operator are computed at every location in the image and are thus segmentation-free. 

One challenge with this approach is that our fragtments are unlabeled. Thus we need some way to relate the textural features of the fragments with the cut-out labeled graphemes. Perhaps we can find a way to generate fake fragments from the labeled graphemes, or we can first do some pre-processing on the fragments as in 

## Bag of Features
In a bag-of-features approach, there are only some features at distinct positions in the image that are used. This approach is also segmentation free. The positions can be identified for example by using a morphological approach or by convolution with some kernel followed by thresholding. Features are computed with respect to their local context. 

Clustering and a codebook can be used to get to a feature vector. 

## Graphemes
We can start with graphemes and normalize their position and size. This way the features can be sensitive to the position in a grapheme where some pattern occurs. Of course this is dependent on first extracting the graphemes from the fragments, which could be difficult. 

**Run-length histograms** in various directions may be a pretty simple feature for style classification.

May use *Text-Independent Writer Identification and Verification Using Textural and Allographic Features*. There contours are represented as a sequence of (x,y) pixel coordinates. This representation is also used to compute the *hinge* feature. Related is a Markov transition probability from angle to angle. 


Also: *Automatic Writer Identification Using Connected-Component Contours and Edge-Based Features of Uppercase Western Script*.

  1. Smoothing
  2. Binarize (midpoint gray value)
  3. Pixel-level contour (Moore's algorithm) starting at leftmost pixel in counter-clockwise fashion
  4. Resample to 100 coordinates
  
# Classification

## Self-organizing time map
[SOTM](https://arxiv.org/abs/1208.1819)
A self-organizing map translates feature vectors into the label of the most similar neuron. During training a map is constructed that triest to preserve the data topology. In a SOTM, the time dimension is another dimension that the map tries to preserve. The map from a previous time is used as an initalization for the next time. 

[MiniSOM](https://github.com/JustGlowing/minisom) seems like a nice library since it is numpy-based and supposedly easy to adapt. It has a bunch of examples and a relatively simple codebase. 

# Data
The original data is in `../../style/`, `../../image-data/` and `../../monkbrill` for me. 

It would be good to figure out how to load these into a skimage.ImageCollection. The goal is to be able to construct a sklearn pipeline that goes from the raw images to the fraglet features. 

For now, simply load the dataset with io.load_collection, binarize, select the largest connected component, compute the contour, resample the contour, rescale the contour. What we want is a list of labels and a list of images. 

# Cross-validation
It would be nice to use sklearn pipeline for this.

# Progress

 - [x] Set up data and branch
 - [x] Load fragments and graphemes
 - [x] Get pixel-level contours
 - [x] Resample contours
 - [x] Get contours and labels in a single array
 - [x] Train miniSOM
 - [ ] Plot miniSOM output
 
 How to plot the MiniSOM output?
 What we need is the centroid of each neuron. Let me start by just plotting the neuron itself. 
 
 Can find contours with `skimage.measure.find_contours`. Each contour is an ndarray of shape (n, 2), consisting of n (row, column) coordinates along the contour. The order of the contours in the output list is determined by the position of the smallest x,y (in lexicographical order) coordinate in the contour. This is a side-effect of how the input array is traversed, but can be relied upon.
 
How exactly do we do this? I think I should create a class containing the components. 



 I can apply transformations to all characters and see the result in the `graphemes` notebook by storing them together in a numpy array. 