import xarray as xr
import numpy as np
import argparse
import os
import fdasrsf.curve_functions as curve_functions
import tqdm
import umap
#import umap.plot
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
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

def read_fraglets(workdir, args):
    fraglet_path = os.path.join(workdir, 'fraglets.nc')
    print('Reading fraglets images from {}'.format(fraglet_path))
    fraglet_data = xr.open_dataset(fraglet_path)
    print('Read {} fraglets.'.format(len(fraglet_data.fraglet_id)))
    # Filter fraglets
    fraglet_data = fraglet_data.where(fraglet_data.area >= args.min_area, drop=True)
    periphery_factor = fraglet_data.periphery / np.sqrt(fraglet_data.area.values) 
    fraglet_data = fraglet_data.where(periphery_factor >= args.min_periphery_factor, drop=True)
    print('Remaining fraglets after filtering: {}.'.format(len(fraglet_data.fraglet_id)))
    return fraglet_data


def normalize_fraglets(fraglet_ds):
    """ Normalize fraglets in the dataset. 
    """
    normlized_contours = xr.zeros_like(fraglet_ds.contour)
    for i in tqdm.trange(len(fraglet_ds.fraglet_id), desc='Normalizing fraglets'):
        beta, q, T = curve_functions.pre_proc_curve(fraglet_ds.contour[i].values.T)
        normlized_contours[i] = beta.T
    fraglet_ds['contour_norm'] = normlized_contours

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Classify fragment style based on labeled fraglets and fragment fraglets')
    parser.add_argument('fragment_workdir', type=check_dir, help='Work directory containing fraglets of the fragments')
    parser.add_argument('codebook_workdir', type=check_dir, help='Work directory containing fraglets of the codebook')
    parser.add_argument('output_dir', type=make_dir, help='Directory where classification results will be written')
    parser.add_argument('-a', '--min_area', type=float, default=100., help='Minimum fraglet area')
    parser.add_argument('-p', '--min_periphery_factor', type=float, default=3.5, help='Minimum fraglet factor periphery / sqrt(Area)')
    args = parser.parse_args()
    
    fragment_fraglets = read_fraglets(args.fragment_workdir, args)
    codebook_fraglets = read_fraglets(args.codebook_workdir, args)
    normalize_fraglets(fragment_fraglets)
    normalize_fraglets(codebook_fraglets)
    
    # Create codebook
    regressor = umap.UMAP()
    codebook_data = codebook_fraglets['contour_norm'].values.reshape(-1,200)
    fragment_fraglets = fragment_fraglets['contour_norm'].values.reshape(-1,200)
    
    codebook_style = codebook_fraglets['style'].values.astype('str')
    # https://umap-learn.readthedocs.io/en/latest/supervised.html?highlight=classification#using-labels-to-separate-classes-supervised-umap
    
    label_encoder = LabelEncoder()
    style_encoder = LabelEncoder()
    style_encoder.fit(codebook_style)
    label = codebook_fraglets['style'].astype('str').str.cat(codebook_fraglets['allograph'].astype('str')).values

    target = label_encoder.fit_transform(label)
    codebook_embedding = regressor.fit_transform(codebook_data, y=target)
    
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(*codebook_embedding.T, s=0.1, c=style_encoder.transform(codebook_style), cmap='Spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title('Supervised UMAP embbedding');
    plt.show()
    
    fragments_embedding = regressor.transform(fragment_fraglets)
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(*codebook_embedding.T, s=0.1, c=style_encoder.transform(codebook_style), alpha=1.0)
    plt.scatter(*fragments_embedding.T, s=0.1, c='b', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title('Supervised UMAP embbedding (codebook and fragments)');
    plt.show()