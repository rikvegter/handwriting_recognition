"""
Load Dead Sea Scrolls image data (fragments and extracted graphemes), and extract contours for use 
in style classification. 

Data will be returned as a pandas.dataframe with related properties (image, labels, filename, centroid pixel)
"""
# scikit-image imports
from skimage.util import img_as_uint, invert
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
import skimage.transform as transform
from skimage.measure import label, find_contours, regionprops
import skimage.io as io

# For drawing Habbakuk font
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import os
import pandas as pd

# Some definitions

# Periods
periods = ["Archaic", "Hasmonean", "Herodian"]

# Allograph names
allographs = ['Alef', 'Bet', 'Gimel', 'Dalet', 'He', 'Waw', 'Zayin', 'Het', 
   'Tet', 'Yod', 'Kaf', 'Kaf-final', 'Lamed',  'Mem-medial', 'Mem',
   'Nun-medial', 'Nun-final', 'Samekh', 'Ayin', 'Pe', 'Pe-final',
   'Tsadi-medial', 'Tsadi-final', 'Qof', 'Resh', 'Shin', 'Taw']

# Keys for Habbakuk font
hab_key = {
    'Alef' : ')',
    'Bet'  : 'b',
    'Gimel' : 'g',
    'Dalet' : 'd',
    'He' : 'h',
    'Waw' : 'w',
    'Zayin' : 'z',
    'Het' : 'x',
    'Tet' : '+',
    'Yod' : 'y',
    'Kaf' : 'k',
    'Kaf-final' : '\\',
    'Lamed' : 'l',
    'Mem-medial' : 'm',
    'Mem' : '{',
    'Nun-medial' : 'n',
    'Nun-final' : '}',
    'Samekh' : 's',
    'Ayin' : '(',
    'Pe' : 'p',
    'Pe-final' : 'v',
    'Tsadi-medial' : 'c',
    'Tsadi-final' : 'j',
    'Qof' : 'q',
    'Resh' : 'r',
    'Shin' : '$',
    'Taw' : 't'
}

def fragments(path = "../../image-data-bin/", sample_points = 100):
    """ Load binary fragments and extract contours
    """
    
    pattern = os.path.join(path, '*')
    files = io.collection.glob(pattern)
    data = []
    for f in files:
        d = pd.DataFrame()
        images, centroids = extract_component_imgs(f)
        print(f, " has components: ", len(images))
        d['img'] = images
        d['centroid'] = centroids
        d['filename'] = f
        d['fragment'] = f
        d['contour'] = d['img'].apply(get_largest_contour, sample_points = sample_points)
        data.append(d)
    data = pd.concat(data, ignore_index=True)
    return data

def extract_component_imgs(file, gauss_r = 3, min_area = 100):
    """ Extract component contours
    """
    img = io.imread(file).astype(np.uint8)
    
    if len(img.shape) == 3:
        img = rgb2gray(img)
    img = gaussian(img, gauss_r)
    img = img > threshold_otsu(img)
    
    # Label the components
    label_image = label(img, background = 1)
    props = regionprops(label_image)
    images = []
    centroids = []
    
    for i, prop in enumerate(props):
        if prop.area > min_area:
            images.append(make_square(invert(prop.image)))
            centroids.append(prop.centroid)
    return images, centroids

def load_graphemes(path, num = None, gauss_r = 3, size = 100, padding = 5, sample_points = 100):
    """ Load grapheme images
    """
    # Construct list of filenames
    d = {'filename' : [], 'allograph' : []}
    for a in allographs:
        pattern = pattern = os.path.join(path, a, '*')
        files = io.collection.glob(pattern)[:num]
        d['filename'].extend(files)
        d['allograph'].extend([a for f in files])
        
    data = pd.DataFrame(d)
    data['img'] = data['filename'].apply(load_with_size, gauss_r=gauss_r, size=size, padding=padding)
    data['contour'] = data['img'].apply(get_largest_contour, sample_points = sample_points)
    return data

def load_with_size(f, size = 100, padding = 5, gauss_r = 2):
    """ Load grayscale image and resize to fixed dimensions
    """
    img = io.imread(f).astype(np.uint8)
    if len(img.shape) == 3:
        img = rgb2gray(img)

    # Make square
    square_size = np.max(img.shape)
    pad_dim = np.maximum(square_size - img.shape, 0)

    # Pad to square image
    pad_before = np.floor(pad_dim / 2.0).astype(np.int)
    pad_after = np.ceil(pad_dim / 2.0).astype(np.int)
    pad_width = ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]))
    img = np.pad(img, pad_width, constant_values = 255)
    # Resize to fixed size
    img = transform.resize(img, (size,size), anti_aliasing=True)
    # Blur
    img = gaussian(img, gauss_r)
    # Binarize
    img = img > threshold_otsu(img)
    # Pad
    img = np.pad(img, ((padding, padding), (padding,padding)), constant_values = True)
    return img

def make_square(img, size = 100, padding = 5):
    square_size = np.max(img.shape)
    pad_dim = np.maximum(square_size - img.shape, 0)

    # Pad to square image
    pad_before = np.floor(pad_dim / 2.0).astype(np.int)
    pad_after = np.ceil(pad_dim / 2.0).astype(np.int)
    pad_width = ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]))
    img = np.pad(img, pad_width, constant_values = 255)
    # Resize to fixed size
    img = transform.resize(img.astype(np.uint8), (size,size), anti_aliasing=True)
    
    img = img > threshold_otsu(img)
        
    img = np.pad(img, ((padding, padding), (padding,padding)), constant_values = True)
    return img
def get_largest_contour(img, sample_points = 100):
    """
    Get the largest contour in the image
    """
    labels = label(invert(img))
    if (labels.max() != 0):
        # There image is not completely empty
        largest_label = np.argmax(np.bincount(labels.flat)[1:])+1
        largest_cc = (labels == largest_label)
        # Get contours
        ct = find_contours(largest_cc, 0.5)
        if len(ct) > 0:
            # There should always be one contour here
            contour = ct[0]
            contour_resampled = np.zeros((2, sample_points), dtype = np.float)
            x = contour[:,0]
            y = contour[:,1]

            contour_resampled[0] = np.interp(
                np.linspace(0, 1, sample_points, endpoint=True),
                np.linspace(0, 1, len(x), endpoint=True), 
                x)
            contour_resampled[1] = np.interp(
                np.linspace(0, 1, sample_points, endpoint=True),
                np.linspace(0, 1, len(y), endpoint=True), 
                y)
            return contour_resampled

def monkbrill(path = "../../monkbrill", **kwargs):
    """ Load monkbrill graphemes
    """
    return load_graphemes(path, **kwargs)

def style_graphemes(path = "../../style", **kwargs):
    """ Load style example graphemes
    """
    data = []
    for s in periods:
        style_path = os.path.join(path, s)
        style_data = load_graphemes(style_path, **kwargs)
        style_data['period'] = s
        data.append(style_data)
    data = pd.concat(data, ignore_index=True)
    return data

def draw_hab(allograph, size = 100, padding = 5, gauss_r = 2):
    ''' Draw a Habbakuk version of the specified allograph
    '''
    hab_font = ImageFont.truetype("../../habbakuk/Habbakuk.TTF", 80)
    # Create black/white image
    img=Image.new("L", (100,100),(255))
    draw = ImageDraw.Draw(img)
    key = hab_key[allograph]
    draw.text((padding, padding),key,(0),font=hab_font)
    img = np.array(img)
    # Blur
    img = gaussian(img, gauss_r)
    # Binarize
    img = img > threshold_otsu(img)
    return img

def habbakuk(path = "../../habbakuk/Habbakuk.TTF", gauss_r = 2, size=100, padding=5, sample_points = 100):
    data = pd.DataFrame({'allograph' : allographs})
    data['img'] = data['allograph'].apply(draw_hab, gauss_r=gauss_r, size=size, padding=padding)
    data['contour'] = data['img'].apply(get_largest_contour, sample_points = sample_points)
    return data

def style_fragments(path = "../../full_images_periods"):
    pass

def split_contours(data):
    data['max_dist_ratio'] = 0.
    for i in range(len(data)):
        c = data.contour[i]
        c_a = c[:,None,:]
        c_b = c[:,:,None]
        straight_dist = np.sqrt(np.sum((c_a - c_b)**2, axis=0))
        a = np.arange(100)[None,:]
        b = np.arange(100)[:,None]
        contour_dist = np.minimum((a - b) % 100, (b - a) % 100)
        contour_dist[straight_dist == 0] = 0
        straight_dist[straight_dist == 0] = 1
        dist_ratio = contour_dist / straight_dist
        data.loc[i, 'max_dist_ratio'] = dist_ratio.max()
# TODO: augmentation by morphing