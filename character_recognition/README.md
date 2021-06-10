# Creating the dataset

For all commands and instructions listed here it is assumed you already have a (virtual) Python environment set up with all requirements installed.

### Building the dataset yourself

If you are on Linux (or on MacOS, if you have the correct packages), you can create the dataset yourself.

0) Ensure you have the `monkbrill.tar.gz` archive in the directory.
1) Ensure you have the correct packages installed. We use gawk, perl (a somewhat recent version is required), find,
   grep, tar, and Bash 4+.
2) Run the `create_dataset` script. This may take a while depending on your computer.

### Downloading the dataset

Alternatively, the dataset we used can be
downloaded [here](https://drive.google.com/file/d/1rbi6yQPT2JjJn0IBdo2_0ZqRc4GO0vNp/view?usp=sharing).

# Training the model

### Training the model yourself
0) Ensure you have the dataset.
1) Run `character_recognizer.py`.

### Downloading a pre-training model
Alternatively, we have made a pre-training model available [here](https://drive.google.com/file/d/1utDs6NSGYzpgvAll2yvAOp1SF-z6ydza/view?usp=sharing).

# Using the character classifier
Once you have obtained a trained model, take note of its path. Then create a new CharacterClassifier object, providing it with the path to the model.

Use `CharacterClassifier#classify_image()` or `CharacterClassifier#classify_images()` depending on whether you have a single or more images. The images can be any of the following types:
   1) A 2d or 3d Numpy array containing a binary, greyscale, or RGB image.
   2) A `PIL.Image.Image` object.
   3) A filename of an image.

The `classify_images()` method accepts lists containing these objects (mixing is allowed), or a numpy array. Note that when supplying a numpy array to this method, it is assumed that the first axis describes the image index.

The classify methods return the index (or indices) of the letter in the LABELS list, which is a list of all letters that can be classified, in reverse alphabetical order.

