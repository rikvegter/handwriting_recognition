import ctypes
import os
from enum import Enum

import numpy as np
import scipy.ndimage as nd
from numpy.core.fromnumeric import clip
from PIL import Image
from scipy import LowLevelCallable


class SegmentationMethod(Enum):
    PROJECTION_PROFILE = 1,
    CONNECTED_COMPONENTS = 2,
    THINNING = 3


class Segmenter:
    """Character segmentation.
    """
    def __init__(self,
                 n_lines: int,
                 labeled_lines: np.ndarray,
                 debug: bool = False,
                 output_path: str = "./") -> None:
        self.debug = debug
        self.output_path = os.path.join(output_path, "characters/")
        self.labeled_lines = labeled_lines
        self.n_lines = n_lines

        if debug:
            os.makedirs(self.output_path, exist_ok=True)

    def segment(self, method: SegmentationMethod):
        for line_no in range(1, self.n_lines + 1):
            line = np.where(self.labeled_lines == line_no, 1,
                            0).astype(np.uint8)

            if method == SegmentationMethod.PROJECTION_PROFILE:
                self.__segment_pp(line_no, line)
            elif method == SegmentationMethod.CONNECTED_COMPONENTS:
                self.__segment_cc(line_no, line)
            elif method == SegmentationMethod.THINNING:
                self.__thin(line_no, line)

    def __segment_pp(self, line_no: int, line: np.ndarray):
        """Character segmentation by projection profiles
        """

        # flip line horizontally so first char is on the right
        line = np.fliplr(line)

        # calculate horizontal projection profile
        projection_profile = np.sum(line, axis=0)
        # print(np.unique(projection_profile))

        # set threshold at percentage of max amount of ink
        # if the pp is below this threshold, cut out a letter
        threshold_percentage = 0.15
        threshold_max = np.max(projection_profile)
        threshold = threshold_percentage * threshold_max

        if threshold_max < 5:
            # if the line is too low, ignore it
            return

        clipped_line = np.where(projection_profile >= threshold,
                                projection_profile, 0)

        labeled_chars, num_chars = nd.label(clipped_line)

        for char_i in range(1, num_chars + 1):
            region_coords = np.argwhere(labeled_chars == char_i)
            min_x = np.min(region_coords)
            max_x = np.max(region_coords)
            if min_x == max_x:
                continue
            char_region = line[:, min_x:max_x]
            # mirror back again
            char = np.fliplr(self.__crop(char_region))

            if self.debug:
                im = Image.fromarray((char * 255).astype(np.uint8))
                im.save(
                    os.path.join(self.output_path,
                                 f"l{line_no}_c{char_i}.png"))

        pass

    def __segment_cc(self, line_no: int, line: np.ndarray):
        """Character segmentation by connected components
        """

        # flip line horizontally so first char is on the right
        line = np.fliplr(line)

        # dilate vertically in order to connect broken lines
        d_struct = np.tile([False, True, False], [3, 1])
        dilated_line = nd.binary_dilation(line, d_struct, iterations=5)

        # detect connected components
        labeled_chars, num_chars = nd.label(dilated_line)

        for char_i in range(1, num_chars + 1):
            # select the current character
            char = np.where(labeled_chars == char_i, 1, 0)
            # "undo" vertical binary dilation
            char = nd.binary_erosion(char, d_struct, iterations=5)
            # if width or height is very small, discard character
            if np.count_nonzero(char) < 20:
                continue
            # remove unnecessary zeroes
            char = self.__crop(char)
            # flip the character back around
            char = np.fliplr(char)

            if self.debug:
                im = Image.fromarray((char * 255).astype(np.uint8))
                im.save(
                    os.path.join(self.output_path,
                                 f"l{line_no}_c{char_i}.png"))

        pass

    def __thin(self, line_no: int, line: np.ndarray):
        """Skeletonization based on Zhang & Suen (1984), doi: 10/c93zqs 

        Method inspired by
        https://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm

        Note: currently applied per line, but could be (a bit) faster when
        applied on the entire image first. This could basically be done with
        something like

        >>> lines # labeled lines 
        >>> binary_lines = np.where(lines, 1, 0) 
        >>> thinned = self.__thin(binary_lines) 
        >>> thinned_lines = thinned * lines
        """

        # define c libs for ndimage general filter
        clib = ctypes.cdll.LoadLibrary("./thinning.so")

        # step one of the algorithm
        clib.StepOne.argtypes = (ctypes.POINTER(
            ctypes.c_double), ctypes.c_long, ctypes.POINTER(ctypes.c_double),
                                 ctypes.c_void_p)
        step_one = LowLevelCallable(clib.StepOne)

        # step 2 of the algorithm
        clib.StepTwo.argtypes = (ctypes.POINTER(
            ctypes.c_double), ctypes.c_long, ctypes.POINTER(ctypes.c_double),
                                 ctypes.c_void_p)
        step_two = LowLevelCallable(clib.StepTwo)

        # If any pixels were set in this round of either step 1 or step 2 then
        # all steps are repeated until no image pixels are changed anymore.
        hasChanged = True
        current = line
        while (hasChanged):
            s1 = nd.generic_filter(current, step_one, size=(3,3))
            s2 = nd.generic_filter(s1, step_two, size=(3,3))

            hasChanged = not np.array_equal(s2, current)
            current = s2

        if self.debug:
            im = Image.fromarray((current * 255).astype(np.uint8))
            im.save(os.path.join(self.output_path, f"l{line_no}_skeleton.png"))

        return current

    def __crop(self, im: np.ndarray) -> np.ndarray:
        """Crop non-zero regions from a numpy array. Basically the inverse of
        padding. Based on https://stackoverflow.com/a/39466129/4545692.

        Args: im (np.ndarray): The array to crop.

        Returns: np.ndarray: The cropped image.
        """
        nonzeroes = np.argwhere(im)
        # top left corner
        tl = nonzeroes.min(axis=0)
        # bottom right corner
        br = nonzeroes.max(axis=0)

        return im[tl[0]:br[0] + 1, tl[1]:br[1] + 1]
