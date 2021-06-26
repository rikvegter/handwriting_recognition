import argparse
import os
import xarray as xr
import skimage.io as io
from itertools import chain, cycle
import tqdm
import numpy as np
from skimage.util import img_as_uint, invert
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
import skimage.transform as transform
from skimage.measure import label, find_contours, regionprops
from skimage.morphology import binary_closing, disk, binary_erosion, opening
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import find_contours
from sklearn.utils import Bunch
from scipy.ndimage.filters import maximum_filter1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d, griddata
from scipy.signal import find_peaks

from xarray_dataclasses import Attr, Coord, Data, datasetclass
from dataclasses import dataclass, field
from typing import Tuple

@datasetclass
class Fraglets:
    """Dataset that represents a collection of fraglets"""
    fraglet_id: Coord['fraglet_id', str]
    contour: Data[Tuple['fraglet_id', 'contour_idx', 'pos'], float]
    style: Data[Tuple['fraglet_id'], str] = field(default_factory = lambda: [''])
    allograph: Data[Tuple['fraglet_id'], str] = field(default_factory = lambda: [''])
    img_id: Data[Tuple['fraglet_id'], str] = field(default_factory = lambda: [''])
    area: Data[Tuple['fraglet_id'], int] = 0
    center: Data[Tuple['fraglet_id', 'pos'], float] = 0
    periphery: Data[Tuple['fraglet_id'], float] = 0
    contour_idx: Coord['contour_idx', int] = field(default=0, init=False)
    pos: Coord['pos', str] = field(default_factory = lambda: ['r', 'c'], init=False)
    def __post_init__(self):
        self.contour_idx = np.arange(self.contour.shape[1])

def check_dir(string):
    """ Check if path exists
    """
    if os.path.isdir(string):
        return os.path.realpath(string)
    else:
        raise NotADirectoryError(string)

def make_dir(string):
    """ Make dir if it does not exist
    """
    if not os.path.isdir(string):
        os.makedirs(string)
    return os.path.realpath(string)

def generate_contours(image):
    """ Generate contours in the image
    """
    for i in range(np.random.randint(0, 5)):
        yield Fraglets.new(
            contour = np.zeros((1,10, 2)), 
            style=['test'], 
            allograph = ['Alef'],
            fraglet_id=['test' + str(i)],
        )
def get_image_components(binary_image, largest_only = False, min_area = 50):
    """ Find connected components in the image and their contours
    
    Find connected components with skimage, and find contours surrounding them. 
    Holes in connected components are filled, and contours run in clockwise direction
    
    Parameters
    ----------
    img_bin : array_like
        Binarized image, as a boolean array with ink as 1 and background as 0
    largest_only : bool
        If True, only return the largest connected component in the image
    min_area : int
        Minimum area in pixels for components to be returned
        
    Returns
    -------
    properties : list of RegionProperties
        Like the output of `skimage.measure.regionprops`, but with an extra `contour` property added
        that holds the contour of the component. 
    """
    label_image = label(binary_image, background = 0)
    props = regionprops(label_image)
    components = []
    
    for i, cc in enumerate(props):
        if cc.area > min_area:

            # We need to pad the image because components should not touch the edge
            cc_img = np.pad(cc.filled_image, 1, constant_values = 0)
            cc_cont = find_contours(cc_img, 0.5, fully_connected="high")
            
            if len(cc_cont) == 0:
                continue
            
            # Smaller contours are holes that are diagonally connected to the 
            # outside, we should only keep the largest contour
            longest_cont_idx = np.argmax([cont.shape[0] for cont in cc_cont])
            # Shift contour to the right position
            longest_cont = cc_cont[longest_cont_idx] + cc.bbox[:2] - [1.,1.]
            cc.contour = longest_cont
            components.append(cc)
    
    if largest_only and len(components) > 1:
        # Only select largest component
        largest_idx = np.argmax([cc.area for cc in components])
        components = [components[largest_idx]]
    return components

def estimate_line_gaps(cc, binary_image, context = (200, 300), line_th = 0.3, gap_th = 2.0):
    """ Estimate the text line gaps overlapping this component
    
    Text line gaps are used to decide where to split the component vertically. Gaps are returned
    if the component touches the densest region of the lines above and below it.
    
    Parameters
    ----------
    cc : RegionProperties
        RegionProperties object describing the component
    binary_image : 2d array of bool
        Binary image
    
    Returns
    -------
    list of pair of int
        For each gap, return a pair of (start, end) rows in coordinates of 
        the full images, containing the index of lowest row of the top line and the 
        highest row of the bottom line.
    """
    
    cc_img = cc.filled_image
    
    # Get image context
    cc_start_r, cc_start_c, cc_end_r, cc_end_c = cc.bbox
    
    # Should look at a fixed-size patch here
    centroid = np.array(cc.centroid)
    
    context_start = np.floor(np.maximum(centroid - context, 0)).astype('int')
    context_end = np.ceil(np.minimum(centroid + context, binary_image.shape)).astype('int')
    
    context = binary_image[
        context_start[0] : context_end[0], 
        context_start[1] : context_end[1]
    ]
    
    context_ink = context.sum(axis=1)
    # Find lines in context
    line = context_ink > context_ink.max() * line_th
    # Connect small gaps in line with 1d morphological closing

    line = binary_closing(line, np.ones(50))
    #line = binary_erosion(line, np.ones(25))
    # Label lines
    line_no = label(line)

    
    # Select line no indices corresponding to the array
    line_r = np.arange(context_start[0], context_end[0])
    cc_line_no = line_no[np.logical_and(
        line_r > cc.bbox[0],
        line_r < cc.bbox[2]
    )]
    
    # Find the labels of the lines touching the component
    cc_line_labels = np.unique(cc_line_no)
    cc_line_labels = np.sort(cc_line_labels[cc_line_labels != 0])
    
    # Find the gaps
    gaps = []
    for start_label, end_label in zip(cc_line_labels, cc_line_labels[1:]):
        gap_start = line_r[np.argwhere(line_no == start_label)[-1].item() + 1]
        gap_end = line_r[np.argwhere(line_no == end_label)[0].item()]
        gaps.append((gap_start, gap_end))
        
    return gaps
    #     plt.imshow(context)
    #     plt.plot(context_ink, np.arange(context.shape[0]))
    #     plt.show()

def find_midline(contour, vertical = False, plot = False):
    """ Returns: 
    top and bot both include start and end extremes along the specified dimension
    """
    
    # Remove duplicate endpoint
    contour = contour[:-1]
    # Row and col coordinates
    r, c = contour.T
    N = r.shape[0]

    if vertical:
        start_idx, end_idx = np.sort([np.argmin(r), np.argmax(r)])
    else:
        start_idx, end_idx = np.sort([np.argmin(c), np.argmax(c)])
    # Split contour
    contour = np.roll(contour, -start_idx, axis=0)
    # Rightmost point should be included in both top and bottom
    top_size = (end_idx + 1) - start_idx
    bot_size = len(contour) - (top_size - 2)
    top = contour[:top_size]
    bot = np.r_[
        contour[1 - bot_size:], 
        top[np.newaxis,0]
    ]
    
    
    # Image bounding box
    min_r, min_c = np.floor(np.min(contour, axis=0)).astype('int')
    max_r, max_c = np.ceil(np.max(contour, axis=0)).astype('int') + 1

    im_shape = (max_r - min_r, max_c - min_c)

    # Coordinates of top and bot in the image
    top_img_ii = np.rint(top - [min_r, min_c]).astype('int')
    bot_img_ii = np.rint(bot - [min_r, min_c]).astype('int')

    top_mask = np.zeros(im_shape, dtype=bool)
    bot_mask = np.zeros(im_shape, dtype=bool)
    top_mask[top_img_ii[:,0], top_img_ii[:,1]] = True
    bot_mask[bot_img_ii[:,0], bot_img_ii[:,1]] = True

    # Store index to find the correspondence of top/bot to midline
    top_idx_mask = np.zeros(im_shape, dtype=int)
    bot_idx_mask = np.zeros(im_shape, dtype=int)
    top_idx_mask[top_img_ii[:,0], top_img_ii[:,1]] = np.arange(top_size)
    bot_idx_mask[bot_img_ii[:,0], bot_img_ii[:,1]] = np.arange(bot_size)
    
    
    dist_top, top_mask_ii = ndimage.distance_transform_edt(~top_mask, return_indices=True)
    dist_bot, bot_mask_ii = ndimage.distance_transform_edt(~bot_mask, return_indices=True)
    
    # Midline will be a contour along the pixes equidistant to top and bottom
    # If points along the contour are not spaced as close as the pixels, there
    # can be a small plateu of zeros, leading to somewhat inaccurate endpoints
    dist_difference = dist_top - dist_bot
    midline_contours = find_contours(dist_difference, 0)

    # There may be multiple contours in some cases, select the longest one
#     try:
    longest_contour_idx = np.argmax([c.shape[0] for c in midline_contours])
#     except ValueError:
#         print("No midline found.")
#         plt.plot(*np.flipud(contour.T))
#         plt.show()
#         plt.figure()
#         im = (dist_difference >= 0.0) * 0.5
#         im += top_mask.astype('int') - 3 * bot_mask
#         #im[midline_ii[:,0], midline_ii[:,1]] = -1
#         plt.imshow(im)
#         plt.show()
    midline_contour = midline_contours[longest_contour_idx]
    midline_ii = np.rint(midline_contour).astype('int')
    
    # TODO: Mask with cc binary image
    midline = midline_contour + (min_r, min_c)
    
    # Correct midline endpoints
    if not vertical:
        midline_mask = np.logical_and(
            midline[:, 1] > top[0,  1], 
            midline[:, 1] < top[-1, 1]
        )
    else:
        midline_mask = np.logical_and(
            midline[:, 0] > top[-1,0], 
            midline[:, 0] < top[0,0]
        )
    
    midline = np.r_[
        top[np.newaxis,0],
        midline[midline_mask], 
        top[np.newaxis,-1]
    ]
    
    midline_ii = np.r_[
        top_img_ii[np.newaxis, 0],
        midline_ii[midline_mask],  
        top_img_ii[np.newaxis, -1]
    ]
    
    if plot:
        plt.figure()
        im = (dist_difference >= 0.0) * 0.5
        im += top_mask.astype('int') - 3 * bot_mask
        #im[midline_ii[:,0], midline_ii[:,1]] = -1
        plt.imshow(im)
    
    
    # These should be equal up to rounding errors
    mid_top_dist = dist_top[midline_ii[:,0], midline_ii[:,1]]
    mid_bot_dist = dist_bot[midline_ii[:,0], midline_ii[:,1]]

    top_idx_ii = top_mask_ii[:, midline_ii[:,0], midline_ii[:,1]]
    bot_idx_ii = bot_mask_ii[:, midline_ii[:,0], midline_ii[:,1]]

    mid_top_idx = top_idx_mask[top_idx_ii[0], top_idx_ii[1]]
    mid_bot_idx = bot_idx_mask[bot_idx_ii[0], bot_idx_ii[1]] 
    
    midline_data = Bunch(
        midline = midline,
        top = top,
        bot = bot, 
        dist = np.minimum(mid_top_dist, mid_bot_dist),
        mid_top_idx = np.maximum.accumulate(mid_top_idx),
        mid_bot_idx = np.minimum.accumulate(mid_bot_idx)
    )
    return midline_data

def midline_substr(data, start, end):
    """ Get contour corresponding to a substring of the midline
    """
    start = min(start, data.midline.shape[0] - 1)
    end = min(end, data.midline.shape[0] - 1)
    top_slice = slice(data.mid_top_idx[start], data.mid_top_idx[end] + 1)
    bot_slice = slice(data.mid_bot_idx[end], data.mid_bot_idx[start] + 1)

    contour = np.concatenate([
        data.top[top_slice], 
        data.bot[bot_slice],
        data.top[data.mid_top_idx[start]][np.newaxis,:]
    ])
    return contour

def midline_area(data):
    r_top_mean = np.r_[0, 0.5 * (data.top[1:,0] + data.top[:-1,0])]
    c_top_diff = np.r_[0, np.diff(data.top[:,1])]
    top_integral = np.cumsum(r_top_mean * c_top_diff)

    r_bot_mean = np.r_[0, 0.5 * (data.bot[1:,0] + data.bot[:-1,0])]
    c_bot_diff = np.r_[0,np.diff(data.bot[:,1])]
    bot_integrand = r_bot_mean * c_bot_diff
    
    bot_integral = -np.cumsum(bot_integrand[::-1])[::-1]
    
    r_cut_mean = 0.5 * (data.top[data.mid_top_idx,0] + data.bot[data.mid_bot_idx,0])
    c_cut_delta = (data.bot[data.mid_bot_idx,1] - data.top[data.mid_top_idx,1])
    
    cut_integral = r_cut_mean * c_cut_delta
    
    midline_area = top_integral[data.mid_top_idx] + bot_integral[data.mid_bot_idx] + cut_integral
    return midline_area

def curve_length(curve):
    """ Integrate curve length
    
    Parameters
    ----------
    curve : ndarray
        Curve should have shape N x ndim
        
    Returns
    -------
    float : Total length of the curve
    """
    return np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))

def curve_area(curve):
    """ Integrate curve area of a closed curve
    
    Parameters
    ----------
    curve : ndarray
        Curve should have shape N x ndim
        
    Returns
    -------
    float : Area enclosed by the curve
    """
    return np.trapz(curve[:,1], curve[:,0])

def split_contour_vertical(cc, line_gaps, min_curve_length = 150, show_vertical = False):
    """ Split contour vertically in each of the gaps. 
    
    Contours are constructed for the areas outside the gap by splitting along the midline. 
    If both these contours are longer than 150, the
    contour is split just below the higher line. Otherwise, the contour is left as-is. 
    
    Parameters
    ----------
    min_curve_length : int
        Minimum length of the curve around a component for it to be considered a 
        potential separate component. Smaller components are never split off.
    
    Returns
    -------
    list of ndarray
        A list of contours, in the image coordinates.
    """
    if len(line_gaps) == 0:
        return [cc.contour]
    
    contour = cc.contour
    split_contours = []
    for gap_start, gap_end in line_gaps:
        midline = find_midline(cc.contour, vertical=True)
        # Select the part of the midline corresponding to the gap
        #print(gap_start, gap_end, midline.midline[:,0])
        midline_start_idx = np.argwhere(midline.midline[:,0] >= gap_end)[-1][0] + 1
        midline_end_idx = np.argwhere(midline.midline[:,0] > gap_start)[-1][0]
        gap_slice = slice(midline_start_idx, midline_end_idx)
        #plt.plot(*np.flipud(midline.midline[gap_slice].T), c='k', ls=':', lw=5)
        
        # Find the part above the top
        below = midline_substr(midline, 0, midline_start_idx)
        above = midline_substr(midline, midline_end_idx, len(midline.midline))
        
        #print(curve_length(below), curve_length(above))
        if curve_length(above) >= min_curve_length and curve_length(below) >= min_curve_length:
            # Split the contour
            split_contours.append(above)
            contour = midline_substr(midline, 0, midline_end_idx)
    split_contours.append(contour)
    # Plot the resulting contours
    #plt.fill(*np.flipud(cc.contour.T), c='grey')
    if show_vertical:
        plt.plot(*np.flipud(midline.midline.T), c='k', ls=':')
        # Alternating hatch pattern
        color = cycle(['grey', 'lightgray'])
        for c in split_contours:
            plt.fill(*np.flipud(c.T), c=next(color), edgecolor='k')
        plt.axis('equal')
        plt.ylim(cc.bbox[2], cc.bbox[0])
        for gap in line_gaps:
            plt.axhspan(*gap, alpha = 0.5, lw=2, ls=':', facecolor='lightgray', edgecolor='k')
        plt.axis('off')
        plt.legend()
        plt.show()
    return split_contours

def reduce_angle(a):
    """ Shift angle by multiples of 2*pi until it fals in the range (-pi,pi)
    """
    return ((a + np.pi) % (2*np.pi)) - np.pi
def reverse_angle(a):
    return reduce_angle(a - np.pi)

class hinge_interpolator(object):
    def __init__(self, contour, wrap = True):
        self.wrap = wrap
        self.contour = contour
        # Arc length
        self.tau = np.r_[
            0,
            np.cumsum(
                np.linalg.norm(np.diff(contour, axis=0),axis=1)
            )]
    def pos(self, tau):
        """ Interpolate position at the given tau
        """
        if self.wrap:
            return np.vstack([
                np.interp(tau, self.tau, self.contour[:,0], period = self.tau[-1]),
                np.interp(tau, self.tau, self.contour[:,1], period = self.tau[-1])
            ]).T
        else:
            # This should work for horizontal lines
            return np.vstack([
                np.interp(tau, self.tau, self.contour[:,0]),
                np.interp(tau, self.tau, self.contour[:,1], 
                          left = self.contour[0, 1] - 1, 
                          right = self.contour[-1,1] + 1),
                          ]).T
    def angle(self, tau, delta_tau):
        """ Calculate angle of vector from pos(tau) to pos(tau + delta_tau)
        """
        distance = self.pos(tau + delta_tau) - self.pos(tau)
        angle = np.arctan2(
            distance[:,0],
            distance[:,1])
        return angle
    def angle_idx(self, idx, delta_tau):
        """ Return angle of idx
        """
        tau = self.tau[idx]
        return self.angle(tau, delta_tau)
def augment_contour(contour, elastic_displacement = 40., elastic_distance = 40., grid_delta = 25., rot_sigma = 10, shear_sigma = 0.1):
    """ Augment the contour by a random affine transformation followed by an elastic morph
    
    Elastic morph is performed by computing a displacement field consisting of spatially correlated 
    Gaussian noise, where elastic_displacement gives the standard deviation in any point, and 
    elastic_distance gives the distance along which points are correlated. Displacement is computed
    on a 2D grid with points that are grid_delta apart, then interpolated with a cubic spline. 
    
    This can in some cases lead to transformations that are not bijective, especially if the elastic
    displacement is larger than the elastic distance, which can be problematic
    
    Parameters
    ----------
    contour : N X 2 array of float
        The contour that should be augmented
    elastic_displacement : float
        Amplitude of the displacement in any given point, 
    elastic_distance : float
        Displacement will be spatially correlated with this distance
    grid_delta : float
        The distance between points of the interpolation grid in any dimension
    rot_sigma : float
        Standard deviation of rotation angle in degrees
    shear_sigma : float
        Standard deviation of the shear factor
    Returns
    -------
    N X 2 array of float
        The augmented contour
    We must make sure that the properties of the augmentation don't depend on the size of the 
    contour. 
    """
    c_mean = np.mean(contour, axis=0)
    contour = contour - c_mean[np.newaxis]
    # Shear angle in degrees
    shear_factor = np.random.normal(0, shear_sigma)
    #shear_factor = 1. / np.tan(np.deg2rad(shear_angle))
    # Apply the shear transformation
    contour = np.stack([
        contour[:,0],
        contour[:,1] + shear_factor * contour[:,0],
    ], axis = 1)
    
    # Rotate the contour
    rotation_deg =  np.random.normal(0, rot_sigma)
    rotation_rad = np.deg2rad(rotation_deg)
    
    contour = np.stack([
        np.cos(rotation_rad) * contour[:,0] - np.sin(rotation_rad) * contour[:,1],
        np.sin(rotation_rad) * contour[:,0] + np.cos(rotation_rad) * contour[:,1]
    ], axis = 1)

    contour += c_mean[np.newaxis]
    buffer = 2 * np.array([elastic_distance, elastic_distance])
    
    grid_start = np.min(contour, axis = 0) - buffer
    grid_end = np.max(contour, axis = 0) + buffer
    
    # Make grid
    r_grid = np.arange(grid_start[0], grid_end[0] + grid_delta, grid_delta)
    c_grid = np.arange(grid_start[1], grid_end[1] + grid_delta, grid_delta)
    
    rr, cc = np.meshgrid(r_grid, c_grid)
    # Shift coordinates by spatially correlated Gaussian noise
    A0 = elastic_displacement * elastic_distance / grid_delta
    sigma = elastic_distance / grid_delta
    
    rr_morphed = (rr + A0 * gaussian_filter(
        np.random.normal(size = rr.shape),
        sigma = sigma)
    ).flatten()
    
    
    cc_morphed = (cc + A0 * gaussian_filter(
        np.random.normal(size = cc.shape), 
        sigma = sigma)
    ).flatten()

    # Interpolation points
    points = np.stack([rr.flatten(), cc.flatten()], axis=1)

    morphed_contour = np.stack(
        [
            griddata(points, rr_morphed, contour, method='cubic'),
            griddata(points, cc_morphed, contour, method='cubic')
        ],
        axis = 1
    )

    return morphed_contour



def split_contour_horizontal(contour, smooth = 2):
    """ Split the contour horizontally
    """
    if smooth > 0:
        contour = gaussian_filter1d(contour, sigma=smooth, mode='wrap', axis=0)
    midline = find_midline(contour)
    mid_size = midline.midline.shape[0]
    idx = np.arange(mid_size)
    top_hinge = hinge_interpolator(
        np.r_[midline.top, midline.bot]
    )
#     bot_hinge = hinge_interpolator(
#         np.r_[midline.bot, midline.top]
#     )
#     mid_hinge = hinge_interpolator(
#         midline.midline,
#         wrap = False
#     )
#     
#     mid_right = mid_hinge.angle_idx(
#         idx, 20
#     )
    top_left = reverse_angle(top_hinge.angle_idx(
        midline.mid_top_idx[idx], -30
    ))
    top_right = top_hinge.angle_idx(
        midline.mid_top_idx[idx], 20
    )
    
#     bot_left = reverse_angle(top_hinge.angle_idx(
#         midline.mid_top_idx[idx], -30
#     ))
    
#     bot_right = bot_hinge.angle_idx(
#         midline.mid_bot_idx[idx], 20
#     )
    
#     mid_to_top = -reduce_angle(mid_right - top_left)
#     bot_hinge = -reduce_angle(bot_right - bot_left)
    top_hinge = -reduce_angle(top_right - top_left)
    
    
    split_criterium = maximum_filter1d(top_hinge, 20)
    # The change in the number of consecutive points of the bottom contour that are 
    # far away from the midline is high where there is a descender, we can use this 
    # as a split criterium, in combination with the top hinge
    bot_contour_loop = -np.r_[0, np.diff(midline.mid_bot_idx)]
    bot_contour_loop[bot_contour_loop < 25] = 0
    split_criterium += bot_contour_loop
    peaks, props = find_peaks(split_criterium, height = np.pi / 3)
    
#     if len(contour) > 300:
#         fig, ax = plt.subplots(3, 1)
#         ax[0].plot(mid_to_top, label = 'mid_to_top')
#         #ax[0].plot(mid_to_top_max, label = 'mid_to_top_max')
#         ax[0].plot(top_hinge, label = 'top_hinge')
#         ax[0].plot(bot_hinge, label = 'bot_hinge')
#         ax[0].axhline(np.pi / 3)
#         for peak in peaks:
#             ax[0].axvline(peak)
#         ax[0].legend()
#         ax[1].fill(*np.flipud(contour.T), c='lightgray', edgecolor='k', lw=1)
#         ax[1].set_ylim(np.max(contour[:,0]), np.min(contour[:,0]))
#         ax[2].plot(np.diff(midline.mid_bot_idx))
#         plt.show()
    fraglets = []
    for start, end in zip(np.r_[0, peaks], np.r_[peaks, mid_size]):
        fraglets.append(midline_substr(midline, start, end))
    # Also include merged contours containing two or three consecutive fragments
    if len(peaks) > 1:
        for start, end in zip(np.r_[0, peaks][::2], np.r_[peaks, mid_size][::2]):
            fraglets.append(midline_substr(midline, start, end))
        for start, end in zip(np.r_[0, peaks][1::2], np.r_[peaks, mid_size][1::2]):
            fraglets.append(midline_substr(midline, start, end))
    if len(peaks) > 2:
        for start, end in zip(np.r_[0, peaks][::3], np.r_[peaks, mid_size][::3]):
            fraglets.append(midline_substr(midline, start, end))
        for start, end in zip(np.r_[0, peaks][1::3], np.r_[peaks, mid_size][1::3]):
            fraglets.append(midline_substr(midline, start, end))
        for start, end in zip(np.r_[0, peaks][2::3], np.r_[peaks, mid_size][2::3]):
            fraglets.append(midline_substr(midline, start, end))
    return fraglets

def extract_fraglets(img_id, binary_image, style, allograph, show_vertical = False, single_graphemes = False, num_points = 100, smooth = 1, augment_factor = 0):
    """ Extract fraglets from a binary image

    Extract fraglets from a preprocessed binary image by performing the following steps:
      - Extract connected components
      - Extract contours of the components
      - Estimate text lines locally by looking at the context of each cc
      - Split contours vertically if they cross lines
      - Split contours horizontally along the midline
    

    Parameters
    ----------
    img_id : str
        String that uniquely identifies the image in the dataset, used to construct
        unique ids for the fraglets.
    img_bin : 2d array of bool
        Binarized image, as a boolean array with ink as 1 and background as 0

    Returns
    -------
    xr.Dataset
        Dataset containind the fraglets, indexed along dimension 'fraglet_id'. 
        Fraglets are stored as unnormalized contours with pixel coordinates relative
        to the source image.
    """
    # Find connected components
    components = get_image_components(binary_image)
    
    fraglets = [] 
    for cc in components:
        if not single_graphemes:
            # Graphemes have not been cropped yet in the dataset, so we must check for components
            # that span two text lines and split them if necessary
            line_gaps = estimate_line_gaps(cc, binary_image)
            horizontal_contours = split_contour_vertical(cc, line_gaps, show_vertical = show_vertical)
        else:
            horizontal_contours = [cc.contour]
        if not augment_factor == 0:
            # Augment the horizontal contours
            num_horizontal = len(horizontal_contours)
            for repeat in range(augment_factor):
                for i in range(num_horizontal):
                    horizontal_contours.append(augment_contour(horizontal_contours[i]))
        for contour in horizontal_contours:
            try:
                fraglets.extend(split_contour_horizontal(contour , smooth))
            except ValueError:
                "Failed to find midline, augmentation might be too high"
    
    # Interpolate fraglets
    num_fraglets = len(fraglets)
    interpolated_fraglets = np.zeros([num_fraglets, num_points, 2])
    # Periphery is the length of the contour, which can be used to 
    # filter the fraglets. 
    fraglet_periphery = np.zeros(num_fraglets)
    fraglet_area = np.zeros(num_fraglets)
    for i, fraglet in enumerate(fraglets):
        r = fraglet[:,0]
        c = fraglet[:,1]
        tau = np.r_[
            0, 
            np.cumsum(
                np.linalg.norm(
                    np.diff(fraglet, axis=0), 
                    axis=1)
            )
        ]
        
        tau_sample = np.linspace(0, tau[-1], num_points, endpoint=True)
        interpolated_fraglets[i,:,0] = np.interp(
            tau_sample,
            tau, 
            r
        )
        interpolated_fraglets[i,:,1] = np.interp(
            tau_sample,
            tau, 
            c
        )
        fraglet_area[i] = curve_area(interpolated_fraglets[i])
        fraglet_periphery[i] = tau[-1]
        
    # Calculate fraglet properties
    fraglet_center = np.median(interpolated_fraglets, axis = 1)
    
    
    # Create fraglets dataset
    fraglet_id = np.char.add(
        np.repeat(str(img_id) + "_", num_fraglets),
        [str(i) for i in range(num_fraglets)]
    )
    fraglets_data = Fraglets.new(
        contour = interpolated_fraglets,
        style=np.repeat(style, num_fraglets), 
        allograph = np.repeat(allograph, num_fraglets),
        img_id = np.repeat(img_id, num_fraglets),
        fraglet_id = fraglet_id,
        area = fraglet_area,
        center = fraglet_center,
        periphery = fraglet_periphery
    )
    return fraglets_data
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read preprocessed images from workdir/images.nc and extract fraglets to workdir/fraglets.nc.')
    parser.add_argument('workdir', type=check_dir, help='Work directory')
    parser.add_argument('-f', '--show_first', type=int, default=0, help='Show plots of final fraglets of first N images')
    parser.add_argument('-v', '--show_vertical', action='store_true', help='Show plots of vertical segmentation results')
    parser.add_argument('-n', '--num_points', type=int, default=100, help='Number of points to use when interpolating fraglet contours')
    parser.add_argument('-s', '--smooth', type=float, default=1, help='Sigma of contour smoothing filter')
    parser.add_argument('-a', '--augment_times', type=int, default=0, help='Number of times to repeat the data by augmentation')
    parser.add_argument('--terminate_after', type=int, default=-1, help='Terminate after first N images (for debugging)')
    parser.add_argument('--split_params', type=str, default=None, help='Parameters for horizontal splitting')
    args = parser.parse_args()
    # TODO add augmentation parameters
    image_data_path = os.path.join(args.workdir, 'images.nc')
    print('Reading preprocessed images from {}'.format(image_data_path))
    image_data = xr.open_dataset(image_data_path)
    print('Read {} images in dataset {}.'.format(len(image_data.img_id), image_data.dataset[0].values))
    
    # Iterate over the images and extract fraglets
    image_fraglets = []
    for i, img_id in enumerate(tqdm.tqdm(image_data.img_id)):
        if args.terminate_after != -1 and i > args.terminate_after:
            break
        # Get image
        data_row = image_data.loc[{'img_id' : img_id}].squeeze()
        
        # Get contours
        image_fraglets.append(
            extract_fraglets(
                img_id.values, 
                data_row.img_bin.values,
                data_row.style.values,
                data_row.allograph.values,
                show_vertical = args.show_vertical,
                single_graphemes = image_data.attrs['single_graphemes'],
                num_points = args.num_points,
                smooth = args.smooth,
                augment_factor = args.augment_times
            )
        )
        if i < args.show_first:
            plt.imshow(invert(data_row.img_bin), cmap='gray')
            plt.title(img_id.values)
            plt.show()
    fraglets_dataset = xr.concat(image_fraglets, dim='fraglet_id')
    # Copy attributes from image dataset
    fraglets_dataset.attrs['name'] = image_data.attrs['name']
    fraglets_dataset.attrs['single_graphemes'] = image_data.attrs['single_graphemes']
    dataset_path = os.path.join(args.workdir, 'fraglets.nc')
    print('Extracted {} fraglets.'.format(len(fraglets_dataset.fraglet_id)))
    print('Writing output to {}'.format(dataset_path))
    fraglets_dataset.to_netcdf(dataset_path, encoding = {
        "contour": {"zlib" : True},
    })