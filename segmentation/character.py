import os
from typing import List

import numpy as np
import scipy.ndimage as nd
import scipy.signal as sig
import utils
from PIL import Image

# General options (parent directory)
from options import GeneralOptions

# Segmentation options (this directory)
from .options import CharacterSegmentationOptions


class CharacterSegmenter:
    """Character segmentation.
    """
    def __init__(self, general_options: GeneralOptions,
                 segment_options: CharacterSegmentationOptions,
                 n_lines: int,
                 labeled_lines: np.ndarray,
                 char_height: float,
                 stroke_width: int) -> None:

        self.info = utils.StepInfoPrinter(n_lines)

        self.debug: bool = general_options.debug
        self.output_path: str = os.path.join(general_options.output_path,
                                             "characters/")
        self.labeled_lines: np.ndarray = labeled_lines
        self.n_lines: int = n_lines
        self.char_height: float = char_height
        self.stroke_width: int = stroke_width

        if general_options.debug:
            os.makedirs(self.output_path, exist_ok=True)

    def segment(self) -> List[List[List[np.ndarray]]]:
        """Segments an image in which the lines are labeled into a collection of
        characters. Characters are grouped by words and lines using lists: this
        method returns a list of lines, where the lines are lists of words, and
        where words are lists of characters. They are indexed according to
        right-to-left reading, so the first character reading from the right is
        result[0][0][0].

        Returns:
            List[List[List[np.ndarray]]]: A list of lines, which are lists of
            words, which are lists of characters.
        """

        print("Step 2: Character segmentation")
        self.info.print(f"Found {self.n_lines} lines.", step=False)

        lines = []

        for line_no in range(1, self.n_lines + 1):
            line = np.where(self.labeled_lines == line_no, 1,
                            0).astype(np.uint8)

            # crop to just the line
            line = self.__crop(line)

            if self.debug:
                im = Image.fromarray((line * 255).astype(np.uint8))
                im.save(
                    os.path.join(self.output_path,
                                 f"l{line_no:02}_before.png"))

            lines.append(self.__segment_line(line_no, line))

        self.info.print_done()

        return lines

    def __segment_line(self, line_no: int, line: np.ndarray) -> List[List[np.ndarray]]:
        """Takes a line and tries to segment it into words.

        Args:
            line_no (int): The line number
            line (np.ndarray): The 2d array containing the line
        """

        # flip line horizontally so first char is on the left
        line = np.fliplr(line)

        # deskew
        line = self.__deskew(line)

        if self.debug:
            im = Image.fromarray((np.fliplr(line) * 255).astype(np.uint8))
            im.save(
                os.path.join(self.output_path, f"l{line_no:02}_deskewed.png"))

        # word separation threshold determined using estimated character height
        word_sep = int(0.05 * self.char_height)

        print("Word sep threshold =", word_sep)

        # horizontally dilate for determining word boundaries
        d_struct = np.tile([0, 1, 0], [3, 1]).T
        dilated_line = nd.binary_dilation(line, d_struct, iterations=word_sep)

        # calculate horizontal projection profile for word separation
        projection_profile = np.sum(dilated_line, axis=0)

        clipped_line = np.where(projection_profile > 0, projection_profile, 0)

        # detect (1D) connected components
        words, n_words = nd.label(clipped_line)

        line_words = []

        # loop over words
        for word_no in range(1, n_words + 1):
            self.info.print(
                f"Processing line {line_no:02}/{self.n_lines:02}, word {word_no:02}/{n_words:02}...",
                step=False
            )
            region_coords = np.argwhere(words == word_no)
            min_x = np.min(region_coords)
            max_x = np.max(region_coords)

            if min_x == max_x:
                continue

            word_region = line[:, min_x:max_x]

            if np.count_nonzero(word_region) < 20:
                continue

            word = self.__crop(word_region)

            if self.debug:
                im = Image.fromarray((word * 255).astype(np.uint8))
                im.save(
                    os.path.join(self.output_path,
                             f"l{line_no:02}_w{word_no:02}.png"))

            word_chars = self.__segment_word(line_no, word_no, word)

            line_words.append(word_chars)

        return line_words

    def __segment_word(self, line_no: int, word_no: int, word: np.ndarray) -> List[np.ndarray]:
        """Takes a word and tries to segment it into characters.

        Args:
            line_no (int): The line number
            word_no (int): The word number
            word (np.ndarray): The 2d array containing the word
        """

        # Perform binary closing to smooth edges a bit and close small holes
        word = nd.binary_closing(word)

        # Set dilation rate
        dilation_rate = 5

        # add some vertical padding to account for vertical dilation
        word = np.pad(word, ((dilation_rate, dilation_rate), (0, 0)))

        # Find connected components
        word_labels, n_components = nd.label(word)

        # Structure for vertical dilation (used later)
        d_struct = np.tile([False, True, False], [3, 1])

        # minimum area is determined by the area of a circle with a diameter
        # equal to the estimated character height / 2, i.e. radius half that
        min_area = np.pi * (self.char_height / 4)**2

        # make a copy of the word to work on
        new_word = np.copy(word)

        # perform vertical dilation on very small elements to connect them with
        # possible close elements they belong to
        for label in range(1, n_components + 1):
            component = np.where(word_labels == label, 1, 0)
            if np.count_nonzero(component) <= min_area:
                # Vertical dilation
                component = nd.binary_dilation(component,
                                               d_struct,
                                               iterations=dilation_rate)
                # Combine dilated element with word using logical or
                new_word = new_word | component

        # close again to connect some loose components
        new_word = nd.binary_closing(new_word)

        # label ccs
        ccs, n_ccs = nd.label(new_word)

        ordered_ccs = {}

        # sort ccs by x coordinate of centre
        for n in range(1, n_ccs + 1):
            cc = np.where(ccs == n, n, 0)
            vpp = np.sum(cc, axis=0)
            nz = np.nonzero(vpp)
            centre_x = np.mean(nz)
            ordered_ccs[centre_x] = cc

        new_ccs = [ v for _, v in sorted(ordered_ccs.items()) ]

        chars = []

        for cc in new_ccs:
            cc = np.where(cc, 1, 0)
            cc = self.__crop(cc)

            split_char = self.__split_ligature(cc)

            # add array of chars to char array
            chars += split_char


        if self.debug:
            for char_no, char in enumerate(chars):
                im = Image.fromarray((char * 255).astype(np.uint8))
                im.save(
                    os.path.join(self.output_path,
                             f"l{line_no:02}_w{word_no:02}_c{char_no + 1:02}.png"))

        return chars


    def __split_ligature(self, cc: np.ndarray) -> List[np.ndarray]:
        """Another attempt at splitting ligatures & otherwise connected characters

        Args:
            cc (np.ndarray): A connected component containing one or more letters

        Returns:
            List[np.ndarray]: A list of characters extracted from the connected component.
        """

        # deskew letter(s)
        cc = self.__deskew(cc)

        # Smooth edges
        bin_sm = sig.medfilt(cc, 5)

        # calculate euclidean distance transform
        edt = nd.distance_transform_edt(bin_sm)

        # calculate vertical projection profile of edt
        vpp = np.sum(edt, axis=0)

        # find valleys
        valls = sig.argrelextrema(vpp, np.less_equal, order=int(self.stroke_width * 2.5))[0]
        l = int(len(vpp) * 0.25)
        r = int(len(vpp) * 0.75)
        valls = valls[valls > l]
        valls = valls[valls < r]

        # create mask
        mask = np.zeros_like(vpp)
        mask[valls] = 1

        # combine close points
        mask = nd.binary_closing(mask)
        val_labs, n_labs = nd.label(mask)

        split_at = []

        for n in range(1, n_labs + 1):
            # find x coords of adjacent splits
            region = np.argwhere(val_labs == n)

            if np.count_nonzero(region) > 1:
                # combine adjacents split points into one point
                new_split = int(np.mean(region))
                split_at.append(new_split)
            else:
                # if there's only one split point, just return the split point
                split_at.append(region[0])

        # split image into chars
        chars = []
        if len(split_at) > 0:
            split_at = [0] + split_at + [len(vpp)]
            for i in range(len(split_at) - 1):
                split_l = int(split_at[i])
                split_r = int(split_at[i + 1])
                char = np.fliplr(bin_sm[:, split_l:split_r])
                chars.append(char)
        else:
            chars = [np.fliplr(bin_sm)]

        return chars

    # Utility functions below

    def __crop(self, im: np.ndarray) -> np.ndarray:
        """Crop non-zero regions from a numpy array. Basically the inverse of
        padding. Based on https://stackoverflow.com/a/39466129/4545692.

        Args: im (np.ndarray): An array to crop.

        Returns: np.ndarray: A cropped image.
        """
        nonzeroes = np.argwhere(im)
        # top left corner
        tl = nonzeroes.min(axis=0)
        # bottom right corner
        br = nonzeroes.max(axis=0)

        return im[tl[0]:br[0] + 1, tl[1]:br[1] + 1]


    def __deskew(self, image: np.ndarray) -> np.ndarray:
        """Deskew a line by trying different skews, see which has the most
        separations, picking that and applying it.

        Args: im (np.ndarray): A line to deskew.

        Returns: np.ndarray: A deskewed line.
        """

        # pad to account for shearing going outside image limits
        h, _ = image.shape
        image = np.pad(image, ((0, 0), (h, h)))

        max_height: int = 0
        max_height_k: float = 0

        # try 40 angles between (roughly) -45 and +45 degrees
        for k in np.arange(-1, 1, 0.05):
            # shearing array as affine transformation
            # (https://en.wikipedia.org/wiki/Transformation_matrix#Shearing)
            # (https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations)
            transform = [[1, 0, 0], [k, 1, 0], [0, 0, 1]]
            sheared_image = nd.affine_transform(image, transform)

            # vertical projection profile
            projprof = np.sum(sheared_image, axis=0)
            # calculate max diff between peaks
            height = np.sum((projprof[1:] - projprof[:-1])**2)
            if height > max_height:
                max_height = height
                max_height_k = k

        # Apply the shearing operation that yielded the straightest vertical linesß
        transform = [[1, 0, 0], [max_height_k, 1, 0], [0, 0, 1]]
        deskewed = nd.affine_transform(image, transform)

        # crop so the image is neat
        deskewed = self.__crop(deskewed)

        return deskewed


