import numpy as np
import pickle
import pandas as pd
from utils import reverse_enum
DEFAULT_MODEL = 'feature_character_recognition/models/svm_model.pkl'

class FeatureExtractor:
    "Feature extraction"
    def __init__(self, model_path: str = DEFAULT_MODEL):
        self.model = self.__load_model(model_path)
        self.INK_THRESHOLD = 20

    def __load_model(self, model_path: str):
        model = pickle.load(open(model_path, 'rb'))

    def __get_height_of_letter(self, vert_profile):
        top = 0
        bottom = 0
        in_line = 0
        #Find the highest position in the picture with ink
        for i in range(0, len(vert_profile)):
            if vert_profile[i] > self.INK_THRESHOLD:
                top = i
                break
        #Find the lowest position in the picture with ink
        for index, item in reverse_enum(vert_profile):
            if item > self.INK_THRESHOLD:
                bottom = index
                break

        height = bottom - top
        return height

    def __reverse_enum(L):
       for index in reversed(range(len(L))):
          yield index, L[index]

    def __get_vertical_projection_profile(self, image):
        width = len(image[0])
        height = len(image)

        vert_profile = []

        for i in range(0, height):
            sum_pixel_value = 0
            for j in range(0, width ):


                try:
                    sum_pixel_value += image[j, i]
                except IndexError as error:
                    continue

            vert_profile.append(sum_pixel_value)

        #Normalize the projection profile
        normalized_vert_profile = [x / width for x in vert_profile]

        #For data reduction purposes, cast the list to integers
        normalized_vert_profile = [int(x) for x in normalized_vert_profile]

        return normalized_vert_profile

    def __get_horizontal_projection_profile(self, image):
        width = len(image[0])
        height = len(image)

        horizon_profile = []

        for i in range(0, width):
            sum_pixel_value = 0
            for j in range(0, height):
                try:
                    sum_pixel_value += image[i, j]
                except IndexError as error:
                    continue
            horizon_profile.append(sum_pixel_value)

        #Normalize the horizon profile
        normalized_horizon_profile = [x / height for x in horizon_profile]

        #For data reduction purposes, cast the list to integers
        normalized_horizon_profile = [int(x) for x in normalized_horizon_profile]

        return normalized_horizon_profile

    def __put_profile_in_middle(self, vert_profile):
        """
        Places the letter in the middle of the vertical projection profile
        :param numpy_array: Vertical projection profile
        :return: The vertical projection centered around the middle of the profile
        """
        top_0 = 0
        bot_0 = 0

        #Find the highest ink row
        for i in range(0, len(vert_profile)):
            if vert_profile[i] >= self.INK_THRESHOLD: break
            top_0 += 1

        #Find the lowest ink row
        for index, item in reverse_enum(vert_profile):
            if item >= self.INK_THRESHOLD: break
            bot_0 += 1

        #The profile is already centered
        if abs(top_0 - bot_0) <= 1:
            return vert_profile

        #Shift the array to center it
        shift_steps = int((top_0 - bot_0) / 2)
        centered_vert_profile = shift(vert_profile, -shift_steps, cval = 0)

        return centered_vert_profile

    def __shift4(self, arr, num, fill_value=0):
        if num >= 0:
            return np.concatenate((np.full(num, fill_value), arr[:-num]))
        else:
            return np.concatenate((arr[-num:], np.full(-num, fill_value)))

    def extract_features(self, image):

        #Calculate features
        vert_profile = self.__get_vertical_projection_profile(image)
        vert_profile = self.__put_profile_in_middle(vert_profile)

        horizon_profile = self.__get_horizontal_projection_profile(image)
        horizon_profile = self.__put_profile_in_middle(horizon_profile)

        height = self.__get_height_of_letter(vert_profile)
        width = self.__get_height_of_letter(horizon_profile)

        #Convert the features to pandas series and combine them
        profile = np.concatenate((vert_profile, horizon_profile), axis = 0)
        profile_series = pd.Series(profile)


        local_features = {'height': height, 'width': width}
        local_features_series = pd.Series(local_features)
        series_to_add = local_features_series.append(profile_series)


        #append the PD series to the dataframe
        column_names = ['height', 'width']
        df = pd.DataFrame(columns = column_names)
        df = df.append(series_to_add, ignore_index = True)

        return df
