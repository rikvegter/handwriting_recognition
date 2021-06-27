import argparse
from operator import invert
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import percentile
from scipy.stats.stats import mode
from skimage.transform.hough_transform import hough_line, hough_line_peaks
import utils
from PIL import Image, ImageOps
from scipy import ndimage

# General options (parent directory)
from options import GeneralOptions

# Segmentation options (this directory)
from .options import LineSegmentationOptions


class LineSegmenter:
    """Implementation of "Handwritten Text Line Segmentation by Shredding Text
    into its Lines" by Anguelos Nicolaou and Basilis Gatos (2009; doi:10/b5wsx6)
    """
    def __init__(self, general_options: GeneralOptions, input_path: str,
                 segment_options: LineSegmentationOptions) -> None:
        """Initialize the line shredder

        Args: image_path (str): the path of the image to segment lines of debug
            (bool, optional): Whether to enable debug mode. Defaults to False.
            output_path (str, optional): Where to save debug images to. Defaults
            to "./".
        """
        self.info = utils.StepInfoPrinter(14)

        print("Step 1: Line segmentation")
        self.info.print("Setting up...")
        # Set debug settings
        self.debug = general_options.debug
        self.output_path = general_options.output_path
        self.im_counter = 0  # for labeling image order
        if self.debug:
            os.makedirs(self.output_path, exist_ok=True)

        # Open the image, convert it to a numpy array and make sure it is binary
        # Also check if the baseline is rotated, and if so, rotate it back
        self.info.print("Straightening baseline...")
        self.image: np.ndarray = self.__prepare_image(input_path)

        # 2.1 Preprocessing

        # Label the connected components
        self.info.print("Labeling connected components...")
        self.components, self.n_components = ndimage.label(self.image)

        # Also find stroke width
        self.info.print("Finding stroke width...")
        self.stroke_width: int = self.__find_stroke_width()

        # Despeckle before finding letter height
        self.info.print("Despeckling...")
        self.__despeckle()

        # Find the letter height and define the blurring window
        self.info.print("Finding letter height...")
        self.letter_height: float = self.__find_letter_height()

        # Blur the image (B(x, y))
        self.info.print("Blurring image...")
        self.blur_width: int = (self.letter_height * 6.0).astype(int)
        self.blur_height: int = (self.letter_height * 0.8).astype(int)
        self.blurred_image = self.__blur_image()

    def shred(self) -> Tuple[int, float, int, np.ndarray]:
        """Labels all connected components in an image as belonging to a line.

        Returns:
            Tuple[int, float, int, np.ndarray]: A 4-tuple containing the number of
            lines, the estimated letter height, stroke-width, and the labeled image.
        """

        # 2.2.1 Tracing line areas (LA(x, y))
        self.info.print("Generating white path traces...")
        line_areas = self.__generate_traces()

        # 2.2.2 Labeling line areas (LLA(x, y))
        self.info.print("Labeling line areas...")
        n_lines, labeled_line_areas = self.__get_lla(line_areas)

        # 2.2.3 Tracing line centers (LC(x, y))
        self.info.print("Generating black path traces...")
        line_centers = self.__generate_traces(invert=True)

        # 2.2.4 Labeling line centers
        self.info.print("Labeling line centers...")
        labeled_line_centers = self.__get_llc(labeled_line_areas, line_centers,
                                              n_lines)

        # 2.3.1 Assigning to line centers
        self.info.print("Assigning characters to line, pass 1...")
        result_line_centers = self.__separate_lines(n_lines,
                                                    labeled_line_centers)

        # 2.3.2 Assigning to line areas
        self.info.print("Assigning characters to line, pass 2...")
        result_line_areas = self.__separate_lines(n_lines, labeled_line_areas)

        # The original paper adds the results from 2.3.2 directly to RES(x,y),
        # but for us it is more convenient to do the adding here
        intermediate_result = result_line_areas + result_line_centers

        # 2.3.3 Assigning remaining pixels
        self.info.print("Assigning remaining pixels...")
        result_final = self.__assign_remaining(n_lines, intermediate_result,
                                               labeled_line_areas)

        self.info.print_done()

        return n_lines, self.letter_height, self.stroke_width, result_final

    def __prepare_image(self, image_path: str) -> np.ndarray:
        # Prepare the image
        image = Image.open(image_path)

        # Make sure we're in grayscale
        image = ImageOps.grayscale(image)

        # Convert to numpy array
        image = np.asarray(image)

        # Convert to binary; 255 is white, 0 is black.
        # We want ones where the image is black
        image = np.where(image < 127, 1, 0)

        # Detect and correct possible baseline rotation using hough transform
        image = self.__straighten(image)

        # Flip the image horizontally so beginning of line is at the left (i.e.
        # closer to 0)
        image = np.fliplr(image)

        if self.debug:
            output = Image.fromarray((image * 255).astype(np.uint8))
            output.save(
                os.path.join(self.output_path,
                             f"{self.im_counter}_binarized_image.png"))
            self.im_counter += 1

        return image

    def __find_letter_height(self) -> float:
        """Finds the average letter height based on the average height of the
        connected components

        Returns:
            float: the average letter height in pixels
        """
        slices = ndimage.find_objects(self.components)
        heights = np.zeros(self.n_components)

        for i, s in enumerate(slices):
            height, _ = self.components[s].shape
            heights[i] = height

        # TODO: check for distribution skew by also calculating median and using
        # that if the skew is too large (maybe)
        return np.mean(heights)

    def __find_stroke_width(self) -> int:
        """Finds the stroke width by continuously eroding the image until the
        less than 20% of the originally white pixels remain. The number of
        iterations of erosions times two is taken as the stroke width.

        Returns:
            int: The estimated stroke width
        """

        struct = ndimage.generate_binary_structure(2, 1)

        im_copy = np.copy(self.image)

        white_pixels_initial = np.sum(im_copy)

        percentage_left = 100

        iterations = 0

        while percentage_left > 20:
            iterations += 1
            im_copy = ndimage.binary_erosion(im_copy, struct)
            n_white_pixels_left = np.sum(im_copy)
            percentage_left = n_white_pixels_left / white_pixels_initial * 100

        return iterations * 2

    def __blur_image(self) -> np.ndarray:
        """Blurs the image with the given blur window.

        Returns:
            np.ndarray: The blurred image
        """

        # Use a uniform filter as a fast blur operator
        # This is different from the paper!
        blurred_image = ndimage.uniform_filter(self.image.astype(np.float64),
                                               size=(self.blur_height,
                                                     self.blur_width))

        if self.debug:
            # increase range so blurred image is visible
            output = np.interp(blurred_image,
                               (blurred_image.min(), blurred_image.max()),
                               (0, 255))
            output = Image.fromarray(output.astype(np.uint8))
            output.save(
                os.path.join(self.output_path,
                             f"{self.im_counter}_binarized_image_blurred.png"))
            self.im_counter += 1

        return blurred_image

    def __generate_traces(self, invert: bool = False) -> np.ndarray:
        """Generates a set of traces.

        Args: invert (bool, optional): Whether to invert the image before
            generating the tracers. Defaults to False.

        Returns: np.ndarray: A binary image containing the tracers.
        """

        # k cannot be higher than the max y, x corresponds to image x
        max_y, max_x = self.blurred_image.shape

        # used for calculating the tracers
        offset = self.blur_height // 2

        # Blurred image, possibly inverted
        if invert:
            source_image = 255 - self.blurred_image
        else:
            source_image = self.blurred_image

        # Precompute tracers. Results in a (x × k) array, where k indicates the
        # kth tracer
        tracers = np.empty_like(self.image)

        # initialize all x == 0 to y
        tracers[:, 0] = np.arange(max_y)

        for x in range(max_x):
            if x > 0:
                prev = tracers[:, x - 1]

                # calculate lhs y and bound it in [0, max_y)
                lhs_y = np.clip(prev + offset, 0, max_y - 1)
                lhs = source_image[lhs_y, x]

                # # calculate rhs y and bound it in [0, max_y)
                rhs_y = np.clip(prev - offset, 0, max_y - 1)
                rhs = source_image[rhs_y, x]

                tracers[:, x] = np.where(lhs > rhs, prev - 1,
                                         np.where(lhs < rhs, prev + 1, prev))

        # Prepare image to save traces to
        traces = np.empty_like(self.image)

        # Calculate tracers (vectorized)
        for x in range(max_x):
            col_k_values = tracers[:, x]
            col_y_values = np.arange(0, max_y)
            if invert:
                traces[:, x] = np.where(np.isin(col_y_values, col_k_values), 1,
                                        0)
            else:
                traces[:, x] = np.where(np.isin(col_y_values, col_k_values), 0,
                                        1)

        # If necessary, save intermediary debug images
        if self.debug:
            # undo horizontal flip
            out_img = np.fliplr(tracers)

            # array of k values
            output_1 = Image.fromarray(
                np.interp(out_img, (out_img.min(), out_img.max()),
                          (0, 255)).astype(np.uint8))

            if invert:
                is_inverted = "_inverted"
            else:
                is_inverted = ""

            output_1.save(
                os.path.join(
                    self.output_path,
                    f"{self.im_counter}_tracer_helper{is_inverted}.png"))
            self.im_counter += 1

            # tracers
            # undo horizontal flip
            out_img2 = np.fliplr(traces)

            output_2 = Image.fromarray(
                (np.interp(out_img2, (out_img2.min(), out_img2.max()),
                           (0, 255))).astype(np.uint8))
            output_2.save(
                os.path.join(self.output_path,
                             f"{self.im_counter}_tracers{is_inverted}.png"))
            self.im_counter += 1

        # Return traces
        return traces

    def __get_lla(self, line_areas: np.ndarray) -> Tuple[int, np.ndarray]:
        """Assigns labels to lines

        Args: line_areas (np.ndarray): The line areas resulting from tracing the
            interline spacing

        Returns: Tuple[int, np.ndarray]: The number of lines and the labeled
            image.
        """

        # find connected components
        labels, n_lines = ndimage.label(line_areas)

        # set minimum pixel count
        min_pix_count = self.letter_height**2

        # find the pixel counts of every labeled image part
        label_sizes = ndimage.labeled_comprehension(line_areas, labels,
                                                    np.arange(1, n_lines + 1),
                                                    np.size, np.uint16, 0,
                                                    False)

        # Mark labels to keep
        # labels should start at 1
        labels_to_keep = np.argwhere(
            label_sizes >= min_pix_count).flatten() + 1

        # Set everything we do not want to keep to 0 and relabel so it is labels are monotonously increasing
        output_labels = np.zeros_like(self.image)

        for i, l in enumerate(labels_to_keep):
            output_labels += np.where(labels == l, i + 1, 0)

        if self.debug:
            # undo horizontal flip
            out_img = np.fliplr(output_labels)

            out_img = np.ma.masked_where(out_img == 0,
                                               out_img)
            cm = plt.get_cmap('turbo', lut=len(labels_to_keep) + 1).copy()
            cm.set_bad(color='black')
            colored_image = cm(out_img)
            Image.fromarray(
                (colored_image[:, :, :3] * 255).astype(np.uint8)).save(
                    os.path.join(self.output_path,
                                 f"{self.im_counter}_line_labels.png"))
            self.im_counter += 1

        return len(labels_to_keep), output_labels

    def __get_llc(self, labeled_line_areas: np.ndarray,
                  line_centers: np.ndarray, n_labels: int) -> np.ndarray:

        llc = labeled_line_areas * line_centers

        if self.debug:
            output_labels = np.ma.masked_where(llc == 0, llc)
            # undo horizontal flip
            out_img = np.fliplr(output_labels)
            cm = plt.get_cmap('turbo', lut=n_labels + 1).copy()
            cm.set_bad(color='black')
            colored_image = cm(out_img)
            Image.fromarray(
                (colored_image[:, :, :3] * 255).astype(np.uint8)).save(
                    os.path.join(self.output_path,
                                 f"{self.im_counter}_line_center_labels.png"))
            self.im_counter += 1

        return llc

    def __separate_lines(self, n_lines: int,
                         labeled_line_centers: np.ndarray) -> np.ndarray:
        """Separates the lines based on the calculated line centers

        Args: n_lines (int): The number of lines that have been found
            labeled_line_centers (np.ndarray): The centers of the text lines

        Returns: np.ndarray: All connected components intersecting the center
            line, labeled with the center line they are intersecting
        """

        # prepare result
        res = np.zeros_like(self.image)

        for label in range(1, n_lines + 1):
            # Find the locations of just the current line
            this_line = np.where(labeled_line_centers == label,
                                 labeled_line_centers, 0)
            # Find the connected components that intersect with the current line
            components_on_line = np.where(self.components * this_line != 0,
                                          self.components, 0)
            # Get the unique components
            unique_components = np.unique(components_on_line)
            # Filter out zeroes
            unique_components = unique_components[unique_components != 0]
            # Label components on the line
            res = np.where(np.isin(self.components, unique_components), label,
                           res)
            # Remove from original set of labeled components LIN(x, y)
            self.components = np.where(
                np.isin(self.components, unique_components), 0,
                self.components)

        if self.debug:
            # Print a colored image to represent component labeling
            output_labels = np.ma.masked_where(res == 0, res)
            # undo horizontal flip
            out_img = np.fliplr(output_labels)
            cm = plt.get_cmap('turbo', lut=n_lines + 1).copy()
            cm.set_bad(color='black')
            colored_image = cm(output_labels)
            Image.fromarray(
                (colored_image[:, :, :3] * 255).astype(np.uint8)).save(
                    os.path.join(
                        self.output_path,
                        f"{self.im_counter}_letter_labels_step_1.png"))
            self.im_counter += 1

        return res

    def __assign_remaining(self, n_lines: int, intermediate_result: np.ndarray,
                           labeled_line_areas: np.ndarray):
        """After steps 2.3.1 and 2.3.2, there might be some unlabeled
        components. This step takes care of them.

        Args: intermediate_result (np.ndarray): The result of the earlier
            labeling steps.
        """

        # We take labeled_line_areas iff LIN(x, y) != AND RES(x, y) == 0.
        # | RES(x, y)   if    LIN(x, y) == 0  OR  RES(x, y) != 0
        # | LLA(x, y)   otherwise
        # The way the paper writes this is a bit convoluted but is equal to the
        # above.
        final_image = np.where(
            self.components == 0,
            # LIN(x, y) == 0; RES(x, y) = whatever
            intermediate_result,
            np.where(
                intermediate_result == 0,
                # LIN(x, y) != 0; RES(x, y) == 0
                labeled_line_areas,
                # LIN(x, y) != 0; RES(x, y) != 0
                intermediate_result))

        # rotate back so output corresponds to input
        final_image = np.fliplr(final_image)

        # reverse line labels so first line has label 1 instead of the highest label
        final_image = np.where(final_image != 0, n_lines - final_image + 1,
                               final_image)

        if self.debug:
            # Print a colored image to represent component labeling
            output_labels = np.ma.masked_where(final_image == 0, final_image)
            cm = plt.get_cmap('turbo', lut=n_lines + 1).copy()
            cm.set_bad(color='black')
            colored_image = cm(output_labels)
            Image.fromarray(
                (colored_image[:, :, :3] * 255).astype(np.uint8)).save(
                    os.path.join(self.output_path,
                                 f"{self.im_counter}_letter_labels_final.png"))
            self.im_counter += 1

        return final_image

    def __straighten(self, image: np.ndarray) -> np.ndarray:
        """Straighten the image so the baselines of the text are as horizontal
        as possible.

        Args:
            image (np.ndarray): An image to straighten

        Returns:
            np.ndarray: The straightened image
        """
        # Allow max 30° rotation
        # (relative to 90° as we're looking at the tangent)
        tested_angles = np.deg2rad(np.linspace(105, 75, 30))

        # Calculate the hough space
        h, theta, d = hough_line(image, theta=tested_angles)

        # Calculate the angles
        _, angles, _ = hough_line_peaks(h, theta, d)

        # Round the angles so we can find the mode
        angles = np.around(angles, decimals=2)

        # Calculate the mode and substract 90 to find the relative rotation
        rotation = np.rad2deg(mode(angles)[0][0]) - 90

        # Rotate the image and return the result
        return ndimage.rotate(image, rotation)

    def __despeckle(self):
        """Connected components based despeckling"""

        # Determine speckle size as function of stroke width
        speckle_size = self.stroke_width

        component_labels = np.arange(1, self.n_components + 1)

        # Check the size of all components and create an array indicating which
        # components should be kept based on their index (1-indexed)
        to_keep = ndimage.labeled_comprehension(
            self.image, self.components, component_labels,
            lambda v: np.sum(v) > speckle_size**2, bool, False)

        labels_to_keep = component_labels[to_keep]

        # map replacements
        replacements = { l : (i + 1 if l in labels_to_keep else 0) for i, l in enumerate(labels_to_keep) }

        new_components = np.copy(self.components)
        # remove unnecessary components
        new_components = np.where(np.isin(new_components, labels_to_keep), new_components, 0)
        # relabel components
        for k,v in replacements.items():
            new_components[self.components == k] = v

        # update image
        self.image = np.where(new_components, 1, 0)
        # update components
        self.components = new_components
        # update n components
        self.n_components = len(labels_to_keep)

        if self.debug:
            output = Image.fromarray((self.image * 255).astype(np.uint8))
            output.save(
                os.path.join(
                    self.output_path,
                    f"{self.im_counter}_binarized_image_despeckled.png"))
            self.im_counter += 1
