import argparse
import os
import xarray as xr
import skimage.io as io
from itertools import chain
import tqdm
import numpy as np
from skimage.util import img_as_uint, invert
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
import skimage.transform as transform
from skimage.measure import label, find_contours, regionprops
from skimage.morphology import binary_closing, disk, binary_opening

# Define names for styles and allographs
# Periods
styles = ["Archaic", "Hasmonean", "Herodian"]

# Allograph names
allographs = ['Alef', 'Bet', 'Gimel', 'Dalet', 'He', 'Waw', 'Zayin', 'Het', 
   'Tet', 'Yod', 'Kaf', 'Kaf-final', 'Lamed',  'Mem-medial', 'Mem',
   'Nun-medial', 'Nun-final', 'Samekh', 'Ayin', 'Pe', 'Pe-final',
   'Tsadi-medial', 'Tsadi-final', 'Qof', 'Resh', 'Shin', 'Taw']

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


def load_image_dataset(path, dataset_name, index_styles, index_allographs, img_pattern):
    """ Load images into an xarray dataset
    """
    def index_paths(dirs, key, values):
        """ Expand path dict for all styles or allographs
        Dirs: a list of dicts
        Key: the key of the variable that is indexed (either allograph or style)
        Values: a list of possible values of this variable
        
        Each entry in dirs will be repeated once for every entry in values, 
        with that value appended to the path, setting the key to that value 
        in the dict, and adding the key to the end of the image id
        """
        dirs = list(chain.from_iterable(
            [{key : val, **d} for val in values] 
        for d in dirs))

        for d in dirs:
            d['dir'] = os.path.join(d['dir'], d[key])
            d['img_id'] = d['img_id'] + "_" + d[key]
        return dirs
    # Create glob paths
    dirs = [{
        'dir' : path, 
        'img_id' : dataset_name, 
        'dataset' : dataset_name
    }]

    # Index styles and allographs if necessary
    if index_styles:
        dirs = index_paths(dirs, 'style', styles)
    if index_allographs:
        dirs = index_paths(dirs, 'allograph', allographs)
    
    # Glob dirs to find images
    def load_img(img_path, idx, img_id, dataset, style = '', allograph = '', dir = ''):
        img_id = "{}_{}".format(img_id,idx)
        img = io.imread(img_path)
        
        img_id_coords = ('img_id', [img_id])
        # Convert to grayscale if necessary
        if len(img.shape) == 3:
            img = rgb2gray(img)
            
        # Construct xarray dataset to represent the loaded image and metadata
        r = img.shape[0]
        c = img.shape[1]
        img_da = xr.DataArray(
            img[np.newaxis], coords=[img_id_coords, ('r', np.arange(r)), ('c', np.arange(c))]
        ).astype('uint8')
        img_shape =  xr.DataArray(
            [[r, c]], coords = [img_id_coords, ('pos', ['r', 'c'])]
        )

        ds = xr.Dataset({
            'img_grayscale' : img_da,
            'img_shape' : img_shape,
            'img_path' : xr.DataArray(img_path, coords = [img_id_coords]).astype('str'),
            'style' : xr.DataArray(style, coords = [img_id_coords]).astype('str'),
            'allograph' : xr.DataArray(allograph, coords = [img_id_coords]).astype('str'),
            'dataset' : xr.DataArray(dataset, coords = [img_id_coords]).astype('str'),
        })
        return ds
    
    # Get results
    image_datasets = []
    for index_dict in tqdm.tqdm(dirs, desc="Loading images", total=len(dirs)):
        pattern = os.path.join(index_dict['dir'], img_pattern)
        files = io.collection.glob(pattern)
        for idx, file in enumerate(files):
            image_datasets.append(
                load_img(file, idx, **index_dict)
            )
    dataset = xr.concat(image_datasets, dim='img_id', fill_value = {'img_grayscale' : 255})
    return dataset

def preprocess_img(img,  gaussian_radius, opening_disk, closing_disk):
    """ Preprocess image with the following steps:
         - Gaussian blur
         - Otsu thresholding
         - Morphological opening
         - Morphological closing
    """
    img = gaussian(img, gaussian_radius)
    img = img >= threshold_otsu(img)
    
    # We could simply exchange opening and closing here to avoid inverting
    img = invert(img)
    img = binary_opening(img, disk(opening_disk))
    img = binary_closing(img, disk(closing_disk))
    return img

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess images from a file tree for fraglet extraction. Preprocessed images with metadata are stored as a NetCDF file in workdir/images.nc for further processing.')
    parser.add_argument('image_dir', type=check_dir, help='Root path for image files')
    parser.add_argument('workdir', type=make_dir, help='Work directory for storing intermediate data and plots')
    parser.add_argument('-s', '--style', action='store_true', help='Traverse style subdirectories and label image data with style')
    parser.add_argument('-l', '--labeled',  action='store_true', help='Traverse allograph subdirectories and label image data with allograph')
    parser.add_argument('-p', '--pattern', type=str, default="*.*", help='Image filename pattern (default *.*)')
    parser.add_argument('-n', '--name', type=str, default='', help="Dataset name (default is the last directory of image_dir)")
    parser.add_argument('-g', '--gaussian_radius', type=float, default=0.5, help="Radius of gaussian blur")
    parser.add_argument('-o', '--opening_disk', type=int, default=3, help="Size of opening disk")
    parser.add_argument('-c', '--closing_disk', type=int, default=1, help="Size of closing disk")

    
    args = parser.parse_args()
    
    print_path = args.image_dir
    if args.style:
        print_path = os.path.join(print_path, '<style>')
    if args.labeled:
        print_path = os.path.join(print_path, '<allograph>')
    print_path = os.path.join(print_path, args.pattern)
    print('Reading images with pattern {}'.format(print_path))

    if args.name == '':
        args.name = os.path.basename(args.image_dir)
    print('Dataset name: {}'.format(args.name))
    
    dataset = load_image_dataset(args.image_dir, args.name, args.style, args.labeled, args.pattern)
    dataset.attrs['name'] = args.name
    dataset.attrs['single_graphemes'] = 1 * args.labeled
    # Preprocess: gaussian filter
    img_bin = xr.zeros_like(dataset.img_grayscale, dtype=bool)
    for i in tqdm.trange(len(dataset.img_id), desc="Preprocessing"):
        img_bin[i] = preprocess_img(
            dataset.img_grayscale[i], 
            args.gaussian_radius, 
            args.opening_disk,
            args.closing_disk
        )
    dataset = dataset.assign(img_bin = img_bin)
    
    dataset_path = os.path.join(args.workdir, 'images.nc')

    print('Writing output to {}'.format(dataset_path))
    dataset.to_netcdf(dataset_path, encoding = {
        "img_grayscale": {"dtype": "uint8", "zlib" : True},
        "img_bin": {"zlib" : True},
    })