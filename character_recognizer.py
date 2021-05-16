import os
import pickle
from os import listdir
from typing import List, Optional

import PIL
import matplotlib.pyplot as plt
import numpy
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 256
"""
Note that changing the number of folds also requires (re)creating the dataset to support it. 
"""
N_FOLDS = 5
DATA_AUGMENTATION_DATASET = False
DATA_AUGMENTATION_LAYERS = True

"""
While the true max image width found in the data is 196 pixels, there are only 16 images that exceed 68 pixels in width.

These 16 images come from the following classes:
- 1 Ayin
- 2 Bet
- 5 Gimel
- 1 He
- 2 Het
- 1 Nun-medial
- 1 Samekh
- 3 Taw

All of these classes have 300 examples, so the loss of information is minimal and arguably not worth the massive 
performance hit incurred when zero-padding all files to ~5.3 times the median width (37 px) for all 5537 files 
(excluding data augmentation).

The value of 68 was chosen because it removes the worst outliers without removing any images from the more sparsely 
populated classes.
"""
MAX_IMG_WIDTH = 68
MAX_IMG_HEIGHT = 77

ENABLE_GPU = True
if ENABLE_GPU:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def load_images_from_directory(data_dir: str) -> np.array:
    """
    Loads all images from a given directory.

    Any files exceeding either `MAX_IMG_WIDTH` or `MAX_IMG_HEIGHT` will be skipped.

    Images that are not provided as grayscale images will be converted to grayscale.

    Lastly, all grayscale images are modified such that they have R+G+B components, so that the images can be used as
    regular images by the network.

    :param data_dir: The directory to read images from.
    :return: A 4d Numpy array of the shape (num_images, MAX_IMG_HEIGHT, MAX_IMG_WIDTH, 3).
    """
    _, _, filenames = next(os.walk(data_dir))
    data = np.zeros((len(filenames), MAX_IMG_HEIGHT, MAX_IMG_WIDTH, 3), dtype=np.uint8)

    skipped = 0
    for idx, file in enumerate(filenames):
        image: PIL.Image.Image = Image.open(data_dir + "/" + file)

        if image.width > MAX_IMG_WIDTH:
            print("Skipping image {} because its width of {} exceeds the maximum width of {}"
                  .format(data_dir + "/" + file, image.width, MAX_IMG_WIDTH))
            skipped += 1
            continue

        # Non-binary images will have a tuple for the image colors instead of an integer value.
        # So, when we encounter this, we binarize the image first.
        if type(image.getcolors()[0][1]) != int:
            image = image.convert('1')

        # noinspection PyTypeChecker
        image_np = np.asarray(image, dtype=np.uint8)

        # Invert and map all values to [0 255], and, considering the images are supposed to be binary,
        # remove all values in-between. E.g. [0, 140, 254, 255] will be mapped to [255, 255, 255, 0]
        # (i.e. everything other than pure white is mapped to white, and pure white is mapped to black).
        # Some images are mapped to [0 1]. These are expanded to [0 255] so all images are the same.
        # The colors are inverted so the features are value 255 instead of 0, which would interfere with 0-padding.
        if np.amax(image_np) > 1:
            scale_fun = lambda x: 255 if x < 255 else 0
        else:
            scale_fun = lambda x: 255 if x == 0 else 0
        image_np = np.vectorize(scale_fun)(image_np)

        # Create a new, 4th dimension with 3 values (RGB).
        # While the current image
        image_np = np.repeat(image_np[..., np.newaxis], 3, -1)
        image_np = image_np.astype(dtype=np.uint8)

        real_idx = idx - skipped
        image_shape = image_np.shape
        # Use height + width offset of half the difference between the max and current height/width.
        # This will cause the glyph to be centered in the 0-padding, which reduces data loss when using
        # certain data augmentation methods (e.g. rotation/zoom).
        off_h = int((MAX_IMG_HEIGHT - image_shape[0]) / 2)
        off_w = int((MAX_IMG_WIDTH - image_shape[1]) / 2)
        data[real_idx, off_h:image_shape[0] + off_h, off_w:image_shape[1] + off_w, 0:image_shape[2]] = image_np

        # Uncomment these two lines to export the images to PNG RGB images. Make sure the output directory exists!
        # rewritten = Image.fromarray(data[real_idx, :, :, :])
        # rewritten.save("rewritten_images/" + file + ".png")

    if skipped > 0:
        data = data[0:-skipped, :, :, :]

    return data


def get_data(data_dir: str, labels: List[str]) -> [np.array, np.array]:
    """
    Gets the data from a directory. If it exists, the pickled file will be loaded.

    If no pickled file exists, the data will be loaded (See load_data_for_label) and then saved as a pickle file for
    future use.

    :param data_dir: The directory to read the data from.
    :param labels: The list of labels in the dataset.
    :return: Two Numpy arrays containing the data/features/examples and the corresponding labels, respectively.
    """
    pickled_file = data_dir + ".pckl"

    if os.path.isfile(pickled_file):
        file = open(pickled_file, 'rb')
        (data, data_labels) = pickle.load(file)
        file.close()
        return data, data_labels

    data_labels: np.array = np.zeros(0)
    data: Optional[np.array] = None
    for idx, label in enumerate(labels):
        label_data = load_images_from_directory(data_dir + "/" + label)
        if data is None:
            data = label_data
        else:
            data = numpy.concatenate((data, label_data), axis=0)

        data_labels = numpy.concatenate((data_labels, np.full(label_data.shape[0], idx)))

    file = open(pickled_file, 'wb')
    pickle.dump((data, data_labels), file)
    file.close()

    return data, data_labels


def create_dataset(examples: np.array, labels: np.array, shuffle: bool = False) -> tf.data.Dataset:
    """
    Creates a Tensorflow dataset from two Numpy arrays.

    :param examples: The array containing the data/examples/features.
    :param labels: The array containing the labels corresponding to the entries in the examples array. Each index
                   in the labels array is linked to the same index in the examples array. As such, these arrays must
                   be of the same length.
    :param shuffle: Whether or not to shuffle the data.
    :return: The newly-created dataset.
    """
    assert examples.shape[0] == labels.shape[0]
    assert len(examples.shape) == 4

    dataset = tf.data.Dataset.from_tensor_slices((examples, labels))
    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def get_kfold_data(data_dir: str, fold: int, labels: List[str]) -> [tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Gets the train, test, and validation datasets.

    :param data_dir: The directory containing the data to load. The data is assumed to be split up into N_FOLDS subdirectories.
    :param fold: The fold id to use for the test dataset. For an N_FOLDS value of 5, this can be any number between 0-4
                 (inclusive). The validation set is the next fold over (wrapping to 0 if the selected fold for the test
                 set is the last one). The training set will use all other folds.
    :param labels: The labels to predict.
    :return: The train, test, and validation datasets.
    """
    fold_validate = (fold + 1) % N_FOLDS

    train: np.array = np.zeros((0, MAX_IMG_HEIGHT, MAX_IMG_WIDTH, 3))
    train_labels: np.array = np.zeros(0, dtype=int)

    for idx in range(N_FOLDS):
        if idx != fold and idx != fold_validate:
            fold_path = data_dir + "/" + str(idx)
            if DATA_AUGMENTATION_DATASET:
                fold_path += "_augmented"
            fold_data, fold_labels = get_data(fold_path, labels)
            train = np.concatenate((train, fold_data), axis=0)
            train_labels = np.concatenate((train_labels, fold_labels), axis=0)

    test, test_labels = get_data(data_dir + "/" + str(fold), labels)
    validate, validate_labels = get_data(data_dir + "/" + str(fold_validate), labels)

    print("Dataset sizes: Train: {}, Test: {}, Validate: {}".format(len(train), len(test), len(validate)))

    test_ds = create_dataset(test, test_labels)
    train_ds = create_dataset(train, train_labels, shuffle=True)
    validate_ds = create_dataset(validate, validate_labels, shuffle=True)

    print("Created datasets!")

    return train_ds, test_ds, validate_ds


def get_labels(data_dir: str) -> List[str]:
    """
    Gets all the labels in the dataset.

    :param data_dir: The directory to search in. This is assumed to be the top-level directory which holds all the partial sets.
    :return: The list of labels.
    """
    search_dir = data_dir + "/0"
    return [folder for folder in listdir(search_dir) if os.path.isdir(search_dir + "/" + folder)]


def get_model(labels) -> tf.keras.models.Model:
    """
    Constructs a new model.

    :param labels: The labels to classify.
    :return: The newly-created model.
    """
    input_shape = (MAX_IMG_HEIGHT, MAX_IMG_WIDTH, 3)

    print("Input shape: {}".format(input_shape))

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(preprocessing.Rescaling(1. / 255))
    if DATA_AUGMENTATION_LAYERS:
        model.add(preprocessing.RandomRotation(factor=(1 / 36)))  # +/-  1/36 * 2pi rad (10 deg)
        model.add(preprocessing.RandomZoom(height_factor=0.2))

    model.add(tf.keras.applications.DenseNet121(input_shape=input_shape, include_top=False, pooling="avg"))

    model.add(layers.Dense(128, activation="tanh")),
    model.add(layers.Dense(len(labels), activation="tanh"))

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model


def run_model(model: tf.keras.models.Model, train: tf.data.Dataset, test: tf.data.Dataset, validate: tf.data.Dataset):
    """
    Trains and tests a model.

    :param model: The model to train and test.
    :param train: The dataset containing the training data.
    :param test: The dataset containing the test data.
    :param validate: The dataset containing the validation data.
    :return: The model's accuracy on the test dataset.
    """
    EPOCHS = 48

    try:
        model.summary()
    except ValueError:
        # No need to do anything.
        # The fit function will fail anyway, and that will throw a 'useful' exception.
        # This one's exception is completely useless.
        print("Failed to print the model summary!")

    history = model.fit(train,
                        verbose=1,
                        validation_data=validate,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=12),
                        )

    y_true = np.zeros(0)
    for _, label in test:
        y_true = np.concatenate((y_true, label.numpy()))

    y_pred = np.argmax(model.predict(test), axis=1)
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    #

    epochs_range = range(len(acc))

    plt.figure(figsize=(9, 9))
    plt.suptitle("Test Accuracy: {}".format(test_acc))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    plt.show()
    print("\nTest_acc: {}\nacc: {}\nval_acc: {}\nloss: {}\nval_loss:{}\n".format(test_acc, acc, val_acc,
                                                                                 loss, val_loss))

    return test_acc


def run_experiment(data_dir: str):
    """
    Runs k-fold (see N_FOLDS) cross-validation using the data from the provided input directory.

    :param data_dir: The directory containing the dataset to use.
    """
    labels = get_labels(data_dir)
    outputs = []
    for fold in range(N_FOLDS):
        train, test, val = get_kfold_data(data_dir, fold, labels)
        model = get_model(labels)
        outputs.append(run_model(model, train, test, val))

    print(f'Average test accuracy: {sum(outputs) / len(outputs):.0%}')
    print("Individual test accuracies: {}".format(outputs))


if __name__ == "__main__":
    run_experiment("dataset")
