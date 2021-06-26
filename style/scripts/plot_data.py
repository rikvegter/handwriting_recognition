import xarray as xr
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
from skimage.util import invert

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

def img_crop_zero(img):
    """ Crop zero areas from a binary image
    """
    # Find nonzero rows and cols
    r_nz = np.nonzero(img.sum(axis = 1))[0]
    c_nz = np.nonzero(img.sum(axis = 0))[0]
    return img[r_nz[0]:r_nz[-1] + 1, c_nz[0]:c_nz[-1] + 1]

def read_fraglets(args):
    fraglet_path = os.path.join(args.workdir, 'fraglets.nc')
    print('Reading fraglets images from {}'.format(fraglet_path))
    fraglet_data = xr.load_dataset(fraglet_path)
    print('Read {} fraglets.'.format(len(fraglet_data.fraglet_id)))
    # Filter fraglets
    fraglet_data = fraglet_data.where(fraglet_data.area >= args.min_area, drop=True)
    periphery_factor = fraglet_data.periphery / np.sqrt(fraglet_data.area.values) 
    fraglet_data = fraglet_data.where(periphery_factor >= args.min_periphery_factor, drop=True)
    print('Remaining fraglets after filtering: {}.'.format(len(fraglet_data.fraglet_id)))
    return fraglet_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Classify fragment style based on labeled fraglets and fragment fraglets')
    parser.add_argument('workdir', type=check_dir, help='Work directory containing fraglets and/or images')
    parser.add_argument('plot_type', type=str, choices = ["images", "fraglets", "segmentation"], help='Type of plot to show')
    parser.add_argument('-r', '--grid_rows', type=int, default=5, help='Height of plot grid')
    parser.add_argument('-c', '--grid_cols', type=int, default=5, help='Width of plot grid')
    parser.add_argument('-d', '--disable_labels', action='store_true', help='Width of plot grid')
    parser.add_argument('-a', '--min_area', type=float, default=100., help='Minimum fraglet area')
    parser.add_argument('-p', '--min_periphery_factor', type=float, default=3.5, help='Minimum fraglet factor periphery / sqrt(Area)')
    args = parser.parse_args()
    
#     faglets_path = os.path.join(args.workdir, 'images.nc')
#     print('Reading fragment fraglets from {}'.format(fragment_faglets_path))
#     fragment_faglets_data = xr.open_dataset(fragment_faglets_path)
    
    if args.plot_type == "images":
        # Plot preprocessed images
        image_data_path = os.path.join(args.workdir, 'images.nc')
        print('Reading preprocessed images from {}'.format(image_data_path))
        image_data = xr.open_dataset(image_data_path)
        print('Read {} images from dataset {}.'.format(len(image_data.img_id), image_data.dataset[0].values))
        if image_data.attrs['single_graphemes']:
            print("Dataset contains extracted graphemes, plotting images in a grid")
            print("Close figure (q on window) to show new sample, Ctrl-C on shell to quit")
            while True:
                fig, axes = plt.subplots(args.grid_rows, args.grid_rows)
                image_idx = np.random.choice(len(image_data.img_bin), args.grid_rows * args.grid_rows)
                for i, ax in zip(image_idx, axes.flat):
                    
                    img = image_data.img_bin[i].values
                    img = invert(img_crop_zero(img))
                    ax.matshow(img, cmap='gray')
                    ax.axis('off')
                    if not args.disable_labels:
                        label = str(image_data.style[i].values) + ' ' + str(image_data.allograph[i].values)
                        ax.set_title(label)
                fig.tight_layout()
                plt.show()
        else:
            print("Close figure (q on window) to show next image, Ctrl-C on shell to quit")
            for i in np.arange(len((image_data.img_bin))):
                img = image_data.img_bin[i].values
                img = invert(img_crop_zero(img))
                plt.matshow(img, cmap='gray')
                plt.axis('off')
                plt.title(image_data.img_path[i].values)
                plt.show()
            
    if args.plot_type == "fraglets":
        # Plot fraglets  
        fraglet_data = read_fraglets(args)
        print("Close figure (q on window) to show new sample, Ctrl-C on shell to quit")
        while True:
            fig, axes = plt.subplots(args.grid_rows, args.grid_rows)
            frag_idx = np.random.choice(len(fraglet_data.fraglet_id), args.grid_rows * args.grid_rows)
            for i, ax in zip(frag_idx, axes.flat):
                contour = fraglet_data.contour[i].values
                r, c = contour.T
                ax.fill(c, -r, c='lightgray', edgecolor='k', lw=2)
                ax.plot(c[0], -r[0], 'o', c='k')
                ax.axis('equal')
                ax.axis('off')
                
                if not args.disable_labels:
                    label = str(fraglet_data.style[i].values) + ' ' + str(fraglet_data.allograph[i].values)
                    ax.set_title(label)
            fig.tight_layout()
            plt.show()

    if args.plot_type == "segmentation":
        # Plot fraglets  
        fraglet_data = read_fraglets(args)
        print("Close figure (q on window) to show next image, Ctrl-C on shell to quit")
        img_ids = np.unique(fraglet_data.img_id.values)
        for img_id in img_ids:
            print(img_id)
            contours = fraglet_data.where(fraglet_data.img_id == img_id, drop=True).contour
            for contour in contours.values:
                r, c = contour.T
                plt.fill(c, -r, c='lightgray', edgecolor='k', lw=1)
                plt.axis('equal')
                plt.axis('off')
            plt.show() 