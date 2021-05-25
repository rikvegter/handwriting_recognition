import numpy as np
from os import listdir
from typing import List, Optional
import os
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy.ndimage.interpolation import shift

N_FOLDS = 5
data_dir = 'data/monkbrill'
WIDTH_IMAGE = 80
HEIGHT_IMAGE = 80
INK_THRESHOLD = 20

# use np.concatenate and np.full by chrisaycock
def shift4(arr, num, fill_value=0):
    if num >= 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))

def put_profile_in_middle(vert_profile):
    """
    Places the letter in the middle of the vertical projection profile
    :param numpy_array: Vertical projection profile
    :return: The vertical projection centered around the middle of the profile
    """
    top_0 = 0
    bot_0 = 0

    for i in range(0, len(vert_profile)):
        if vert_profile[i] >= INK_THRESHOLD: break
        top_0 += 1

    for index, item in reverse_enum(vert_profile):
        if item >= INK_THRESHOLD: break
        bot_0 += 1

    #The profile is already centered
    if abs(top_0 - bot_0) <= 1:
        return vert_profile

    #Shift the array to center it
    shift_steps = int((top_0 - bot_0) / 2)
    centered_vert_profile = shift(vert_profile, -shift_steps, cval = 0)

    return centered_vert_profile

def get_labels(data_dir):
    """
    Gets all the labels in the dataset.
    :param data_dir: The directory to search in. This is assumed to be the top-level directory which holds all the partial sets.
    :return: The list of labels.
    """
    search_dir = data_dir
    return [folder for folder in listdir(search_dir) if os.path.isdir(search_dir + "/" + folder)]

#Returns the summed pixel values per pixel height
def get_vertical_projection_profile(image):
    width, height = image.size
    vert_profile = []
    for i in range(0, height):
        sum_pixel_value = 0
        for j in range(0, width):
            if isinstance(image.getpixel((j, i)), (tuple)):
                continue
            sum_pixel_value += image.getpixel((j, i))
        vert_profile.append(sum_pixel_value)

    #Normalize the projection profile
    normalized_vert_profile = [x / image.size[0] for x in vert_profile]

    #For data reduction purposes, cast the list to integers
    normalized_vert_profile = [int(x) for x in normalized_vert_profile]

    return normalized_vert_profile

def reverse_enum(L):
   for index in reversed(range(len(L))):
      yield index, L[index]

def get_height_of_letter(vert_profile):
    top = 0
    bottom = 0
    in_line = 0
    #Find the highest position in the picture with ink
    for i in range(0, len(vert_profile)):
        if vert_profile[i] > INK_THRESHOLD:
            top = i
            break
    #Find the lowest position in the picture with ink
    for index, item in reverse_enum(vert_profile):
        if item > INK_THRESHOLD:
            bottom = index
            break

    height = bottom - top
    return height
#Reads an image an resizes it to the specified values.
def read_img(file_name):
    im = Image.open(file_name)
    #Resize the image
    resized_im = im.resize((WIDTH_IMAGE, HEIGHT_IMAGE))

    resized_invert_im = ImageOps.invert(resized_im)
    return resized_invert_im

def main():
    labels = get_labels(data_dir)
    column_names = ['height', 'label']
    df = pd.DataFrame(columns = column_names)

    for letter in labels:
        file_names = os.listdir(data_dir + '/' + letter)
        for file in file_names:
            image_name = data_dir + '/' + letter + '/' + file
            im = read_img(image_name)

            #Calculate features
            vert_profile = get_vertical_projection_profile(im)
            vert_profile = put_profile_in_middle(vert_profile)
            height = get_height_of_letter(vert_profile)

            #Convert the features to pandas series and combine
            vert_profile_series = pd.Series(vert_profile)
            local_features = {'height': height, 'label': letter}
            local_features_series = pd.Series(local_features)
            series_to_add = local_features_series.append(vert_profile_series)

            #append the PD series to the dataframe
            df = df.append(series_to_add, ignore_index = True)


    #Save the data
    df.to_pickle('local_features.pkl')


if __name__ == "__main__":
    main()
