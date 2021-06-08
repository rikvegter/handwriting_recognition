import ctypes
import os
from typing import Tuple

import numpy as np
import scipy.ndimage as nd
import utils
from PIL import Image
from scipy import LowLevelCallable
from skimage.morphology import convex_hull_image

# General options (parent directory)
from options import GeneralOptions

from .character_segmentation_method import CharacterSegmentationMethod
# Segmentation options (this directory)
from .options import CharacterSegmentationOptions, SegmentationOptions


class CharacterSegmenter:
    """Character segmentation.
    """
    def __init__(self,
                 general_options: GeneralOptions,
                 segment_options: CharacterSegmentationOptions,
                 n_lines: int,
                 labeled_lines: np.ndarray,
                 char_height: float) -> None:
        
        self.debug: bool = general_options.debug
        self.output_path: str = os.path.join(general_options.output_path, "characters/")
        self.labeled_lines: np.ndarray = labeled_lines
        self.n_lines: int = n_lines
        self.char_height: float = char_height

        if general_options.debug:
            os.makedirs(self.output_path, exist_ok=True)

    def segment(self):

        #TODO: Make methods return ragged arrays based on this:
        #      https://tonysyu.github.io/ragged-arrays.html
        #      Alternatively, pad arrays to a given size and return that

        print("Step 2: Character segmentation")
        utils.print_info(f"Found {self.n_lines} lines.")

        for line_no in range(1, self.n_lines + 1):
            line = np.where(self.labeled_lines == line_no, 1,
                            0).astype(np.uint8)

            # 1px erosion to get rid of some noise, followed by dilation to
            # ‘correct’ for the erosion
            # TODO fix this, generates weird output
            # struct = nd.generate_binary_structure(2, 1)
            # line = nd.binary_erosion(line, struct)
            # line = nd.binary_dilation(line, struct)

            # crop to just the line
            line = self.__crop(line)

            if self.debug:
                im = Image.fromarray((line * 255).astype(np.uint8))
                im.save(os.path.join(self.output_path, f"l{line_no:02}_before.png"))

            self.__segment(line_no, line)

        utils.print_info("        Done.")


    def __segment(self, line_no, line):

        # flip line horizontally so first char is on the left
        line = np.fliplr(line)

        # deskew
        line = self.__deskew(line)

        if self.debug:
            im = Image.fromarray((np.fliplr(line) * 255).astype(np.uint8))
            im.save(
                os.path.join(self.output_path,
                                f"l{line_no:02}_deskewed.png"))

        # word separation threshold determined using estimated character height
        word_sep = int(0.1 * self.char_height)

        # horizontally dilate for determining word boundaries
        d_struct = np.tile([False, True, False], [3, 1]).T
        dilated_line = nd.binary_dilation(line, d_struct, iterations=word_sep)

        # calculate horizontal projection profile for word separation
        projection_profile = np.sum(dilated_line, axis=0)

        clipped_line = np.where(projection_profile > 0, projection_profile, 0)

        # detect (1D) connected components
        words, n_words = nd.label(clipped_line)

        # loop over words
        for word_i in range(1, n_words + 1):
            utils.print_info(f"Processing line {line_no:02}/{self.n_lines:02}, word {word_i:02}/{n_words:02}...")
            region_coords = np.argwhere(words == word_i)
            min_x = np.min(region_coords)
            max_x = np.max(region_coords)

            if min_x == max_x:
                continue

            word_region = line[:, min_x:max_x]

            if np.count_nonzero(word_region) < 20:
                continue

            word = np.fliplr(self.__crop(word_region))

            if self.debug:
                im = Image.fromarray((word * 255).astype(np.uint8))
                im.save(
                    os.path.join(self.output_path,
                                 f"l{line_no:02}_w{word_i:02}.png"))

        


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
                                 f"l{line_no:02}_c{char_i:03}.png"))

        pass

    def __segment_cc(self, line_no: int, line: np.ndarray):
        """Character segmentation by connected components
        """

        # Prepare for configurable dilation rate
        dilation_rate = 3

        # flip line horizontally so first char is on the right
        line = np.fliplr(line)

        # dilate vertically in order to connect broken lines
        d_struct = np.tile([False, True, False], [3, 1])
        dilated_line = nd.binary_dilation(line, d_struct, iterations=dilation_rate)

        # # detect connected components
        labeled_chars, num_chars = nd.label(dilated_line)

        for char_i in range(1, num_chars + 1):
            # select the current character
            dilated_char = np.where(labeled_chars == char_i, 1, 0)

            # calculate convex hull
            hull = convex_hull_image(dilated_char)

            # use convex hull to select character from original line
            char = line * hull

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
                                 f"l{line_no:02}_c{char_i:03}.png"))

        pass

    def __thin(self, line_no: int, line: np.ndarray):
        """Skeletonization based on Zhang & Suen (1984), doi: 10/c93zqs 

        Method inspired by
        https://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm

        Note: currently applied per line, but could be (a bit) faster when
        applied on the entire image first. This could basically be done with
        something like

        Note 2: This doesn't work as it is supposed to.

        >>> lines # labeled lines 
        >>> binary_lines = np.where(lines, 1, 0) 
        >>> thinned = self.__thin(binary_lines) 
        >>> thinned_lines = thinned * lines
        """

        if self.debug:
            im = Image.fromarray((line * 255).astype(np.uint8))
            im.save(
                os.path.join(self.output_path,
                             f"l{line_no}_before_thinning.png"))

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
        iterations = 0
        max_iterations = 20
        while (hasChanged and iterations < max_iterations):
            s1 = nd.generic_filter(current,
                                   step_one,
                                   size=(3, 3),
                                   mode="constant",
                                   cval=0)
            s2 = nd.generic_filter(s1,
                                   step_two,
                                   size=(3, 3),
                                   mode="constant",
                                   cval=0)

            hasChanged = not np.array_equal(s2, current)
            current = s2
            iterations += 1

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

    def __deskew(self, im: np.ndarray) -> np.ndarray:
        # try different skews, see which has the most separations, pick that and
        # apply it

        # pad to account for shearing going outside image limits
        h, _ = im.shape
        im = np.pad(im, ((0,0), (h,h)))

        max_zeroes: int = 0
        max_zeroes_k: float = 0

        # try angles between -45 and +45 degrees
        for k in np.arange(-1, 1, 0.1):
            # shearing array
            transform = [[1, 0, 0],
                         [k, 1, 0],
                         [0, 0, 1]]
            sheared_image = nd.affine_transform(
                im, 
                transform
            )

            # vertical projection profile
            projprof = np.sum(sheared_image, axis=0)
            zeroes = np.count_nonzero(projprof == 0)
            if zeroes > max_zeroes:
                max_zeroes = zeroes
                max_zeroes_k = k
            
        # for now just use max height
        transform = [[1,            0, 0], 
                     [max_zeroes_k, 1, 0],
                     [0,            0, 1]]
        return nd.affine_transform(
            im, 
            transform,
        )