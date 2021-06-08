import os
import pickle
from os import listdir
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy
import numpy as np
import seaborn as sn
import tensorflow as tf
from PIL import Image
from matplotlib.axes import Axes
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.callbacks import History

BATCH_SIZE: int = 64
SHUFFLE_BUFFER_SIZE: int = 256
"""
Note that changing the number of folds also requires (re)creating the dataset to support it. 
"""
N_FOLDS: int = 5
DATA_AUGMENTATION_DATASET: bool = True
DATA_AUGMENTATION_LAYERS: bool = True
EPOCHS: int = 96
MODEL_OUTPUT_PATH = "models"
NETWORK_APPLICATION = tf.keras.applications.DenseNet121

IMG_WIDTH: int = 64
IMG_HEIGHT: int = 64

ENABLE_GPU: bool = True
if ENABLE_GPU:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def load_images_from_directory(data_dir: str) -> np.ndarray:
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
    data: np.ndarray = np.zeros((len(filenames), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    skipped: int = 0
    for idx, file in enumerate(filenames):
        image: Image.Image = Image.open(data_dir + "/" + file)
        if image.mode != "L":  # Avoid making a copy
            image: Image.Image = image.convert(mode="L")
        assert image.height == IMG_HEIGHT and image.width == IMG_WIDTH

        # noinspection PyTypeChecker
        image_np: np.ndarray = np.asarray(image, dtype=np.uint8)

        # Create a new dimension of size 3 (RGB) where each channel copies
        # the greyscale value. Image-based stuff usually doesn't handle greyscale very well.
        image_np: np.ndarray = np.repeat(image_np[..., np.newaxis], 3, -1)
        image_np: np.ndarray = image_np.astype(dtype=np.uint8)
        # The colors are inverted so the features have non-0 values, to avoid issues with 0-padding.
        image_np: np.ndarray = 255 - image_np

        real_idx: int = idx - skipped
        image_shape: Tuple[int, int] = image_np.shape
        # Use height + width offset of half the difference between the max and current height/width.
        # This will cause the glyph to be centered in the 0-padding, which reduces data loss when using
        # certain data augmentation methods (e.g. rotation/zoom).
        off_h: int = int((IMG_HEIGHT - image_shape[0]) / 2)
        off_w: int = int((IMG_WIDTH - image_shape[1]) / 2)
        data[real_idx, off_h:image_shape[0] + off_h, off_w:image_shape[1] + off_w, 0:image_shape[2]] = image_np

        # # Uncomment these two lines to export the images as they will be used for the network. Make sure the output directory exists!
        # rewritten = Image.fromarray(data[real_idx, :, :, :])
        # rewritten.save("rewritten_images/" + file + ".pgm")

    if skipped > 0:
        data: np.ndarray = data[0:-skipped, :, :, :]

    return data


def get_data(data_dir: str, labels: List[str]) -> [np.ndarray, np.ndarray]:
    """
    Gets the data from a directory. If it exists, the pickled file will be loaded.

    If no pickled file exists, the data will be loaded (See load_data_for_label) and then saved as a pickle file for
    future use.

    :param data_dir: The directory to read the data from.
    :param labels: The list of labels in the dataset.
    :return: Two Numpy arrays containing the data/features/examples and the corresponding labels, respectively.
    """
    pickled_file: str = data_dir + ".pckl"

    if os.path.isfile(pickled_file):
        file = open(pickled_file, 'rb')
        (data, data_labels) = pickle.load(file)
        file.close()
        return data, data_labels

    data_labels: np.ndarray = np.zeros(0)
    data: Optional[np.ndarray] = None
    for idx, label in enumerate(labels):
        label_data: np.ndarray = load_images_from_directory(data_dir + "/" + label)
        if data is None:
            data = label_data
        else:
            data = numpy.concatenate((data, label_data), axis=0)

        data_labels: np.ndarray = numpy.concatenate((data_labels, np.full(label_data.shape[0], idx)))

    file = open(pickled_file, 'wb')
    pickle.dump((data, data_labels), file)
    file.close()

    return data, data_labels


def create_dataset(examples: np.ndarray, labels: np.ndarray, shuffle: bool = False) -> tf.data.Dataset:
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

    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((examples, labels))
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
    fold_validate: float = (fold + 1) % N_FOLDS

    train: np.ndarray = np.zeros((0, IMG_HEIGHT, IMG_WIDTH, 3))
    train_labels: np.ndarray = np.zeros(0, dtype=int)

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

    test_ds: tf.data.Dataset = create_dataset(test, test_labels)
    train_ds: tf.data.Dataset = create_dataset(train, train_labels, shuffle=True)
    validate_ds: tf.data.Dataset = create_dataset(validate, validate_labels, shuffle=True)

    print("Created datasets!")

    return train_ds, test_ds, validate_ds


def get_labels(data_dir: str) -> List[str]:
    """
    Gets all the labels in the dataset.

    :param data_dir: The directory to search in. This is assumed to be the top-level directory which holds all the partial sets.
    :return: The list of labels.
    """
    search_dir: str = data_dir + "/0"
    return [folder for folder in listdir(search_dir) if os.path.isdir(search_dir + "/" + folder)]


def get_model(labels: List[str]) -> tf.keras.Sequential:
    """
    Constructs a new model.

    :param labels: The labels to classify.
    :return: The newly-created model.
    """
    input_shape: Tuple[int, int, int] = (IMG_HEIGHT, IMG_WIDTH, 3)

    print("Input shape: {}".format(input_shape))

    model: tf.keras.Sequential = tf.keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(preprocessing.Rescaling(1. / 255))
    if DATA_AUGMENTATION_LAYERS:
        model.add(preprocessing.RandomRotation(factor=(1 / 36)))  # +/-  1/36 * 2pi rad (10 deg)
        model.add(preprocessing.RandomZoom(height_factor=0.2))

    # Use max pooling, because the images are inverted such that ink (=255) = white, black (=0) = background.
    model.add(NETWORK_APPLICATION(input_shape=input_shape, include_top=False, pooling="max", weights=None,
                                  classes=len(labels)))

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model


def run_model(model: tf.keras.models.Model, train: tf.data.Dataset, test: tf.data.Dataset, validate: tf.data.Dataset,
              labels: List[str], model_checkpoint_path: Optional[str] = None) -> float:
    """
    Trains and tests a model.

    :param model: The model to train and test.
    :param train: The dataset containing the training data.
    :param test: The dataset containing the test data.
    :param validate: The dataset containing the validation data.
    :param labels: The list of labels.
    :param model_checkpoint_path: The path to use for storing model checkpoints. When this value is not provided,
    the model will not be stored.
    :return: The model's accuracy on the test dataset in [0, 1].
    """
    try:
        model.summary()
    except ValueError:
        # No need to do anything.
        # The fit function will fail anyway, and that will throw a 'useful' exception.
        # This one's exception is completely useless.
        print("Failed to print the model summary!")

    if model_checkpoint_path is None:
        save_model_callback = None
    else:
        save_model_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path,
            save_best_only=True,
            monitor="val_loss", mode="min",
            save_weights_only=True
        )

    history: History = model.fit(train,
                                 verbose=1,
                                 validation_data=validate,
                                 epochs=EPOCHS,
                                 batch_size=BATCH_SIZE,
                                 callbacks=[save_model_callback]
                                 )

    if model_checkpoint_path is not None and save_model_callback is not None:
        print("Loading optimal weights...")
        model.load_weights(model_checkpoint_path)

    y_true: np.ndarray = np.zeros(0)
    for _, label in test:
        y_true = np.concatenate((y_true, label.numpy()))

    y_pred: np.ndarray = np.argmax(model.predict(test), axis=1)
    test_acc: float = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    acc: List[float] = history.history["accuracy"]
    val_acc: List[float] = history.history["val_accuracy"]

    loss: List[float] = history.history["loss"]
    val_loss: List[float] = history.history["val_loss"]

    #

    ax: Axes = sn.heatmap(confusion_matrix(y_true, y_pred), annot=False, xticklabels=labels, yticklabels=labels)

    # Use our own x/y ticks on full integers. The default x/y ticks go on halves, meaning that the grid would
    # go through the center of each square in the heatmap, which is just confusing.
    # We use minor ticks to avoid shifting the x/y labels (which are attached to the major ticks).
    ax.set_xticks(range(0, len(labels)), minor=True)
    ax.set_yticks(range(0, len(labels)), minor=True)
    ax.grid(b=True, which='minor', axis='both')

    # Get rid of the annoying minor ticks on the x/y axes that would appear there because we're using the minor ticks.
    for tic in ax.xaxis.get_minor_ticks():
        tic.tick1line.set_visible(False)
    for tic in ax.yaxis.get_minor_ticks():
        tic.tick1line.set_visible(False)

    ax.set_title("Confusion Matrix of Character Recognizer")
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")
    ax.figure.tight_layout()

    plt.show()

    #

    epochs_range: range = range(len(acc))

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


def train_model(data_dir: str, model_output_path: Optional[str] = None, results_output_path: Optional[str] = None):
    """
    Runs k-fold (see N_FOLDS) cross-validation using the data from the provided input directory.

    :param data_dir: The directory containing the dataset to use.
    :param model_output_path: The directory to store models in.
    :param results_output_path: The file to store the results in. If the file already exists, it will be overwritten.
    """
    labels: List[str] = get_labels(data_dir)
    outputs: List[float] = []
    for fold in range(N_FOLDS):
        train, test, val = get_kfold_data(data_dir, fold, labels)
        model: tf.keras.Sequential = get_model(labels)
        outputs.append(run_model(model, train, test, val, labels, "{}/model_{}/".format(model_output_path, fold)))
        # break

    output: str = f'Average test accuracy: {sum(outputs) / len(outputs):.0%}\nIndividual test accuracies: {outputs}'
    print(output)

    if results_output_path is not None:
        with open(results_output_path, "w") as file:
            file.write(output)


if __name__ == "__main__":
    if not os.path.isdir(MODEL_OUTPUT_PATH):
        os.mkdir(MODEL_OUTPUT_PATH)
    train_model("dataset_preprocessed_2ximgmorph_shear_dilation_erosion", MODEL_OUTPUT_PATH, "Results")
