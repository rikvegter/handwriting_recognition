"""
Train a style classifier, write performance metrics and write the classifier to disk
"""

import xarray as xr
import numpy as np
import argparse
import os
import tqdm
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn import preprocessing, cluster, feature_selection
import codecs
import pickle

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
    """ Read fraglets form disk into an xarray dataset, and filter based on area and peripheriy factor
    """
    fraglet_path = os.path.join(workdir, 'fraglets.nc')
    print('Reading fraglets images from {}'.format(fraglet_path))
    fraglet_data = xr.load_dataset(fraglet_path)
    print('Read {} fraglets.'.format(len(fraglet_data.fraglet_id)))
    # Filter fraglets
    fraglet_data = fraglet_data.where(fraglet_data.area >= args.min_area, drop=True)
    periphery_factor = fraglet_data.periphery / np.sqrt(fraglet_data.area.values) 
    fraglet_data = fraglet_data.where(periphery_factor >= args.min_periphery_factor, drop=True)
    print('Remaining fraglets after filtering: {}.'.format(len(fraglet_data.fraglet_id)))
    return fraglet_data


def normalize_fraglets(fraglet_ds):
    """ Normalize fraglets in the dataset. 
    
    Dataset is modified in-place by adding two new variables (contour_norm and square_root_velocity)
    
    Normalize by:
      - Subtracting mean
      - Setting standard deviation to one
    """
    normlized_contours = xr.zeros_like(fraglet_ds.contour)
#     sqare_root_velocity = xr.zeros_like(fraglet_ds.contour)
    normlized_contours.values[:,:,0] = preprocessing.scale(
        fraglet_ds.contour.values[:,:,0],
        axis=1
    )
    normlized_contours.values[:,:,1] = preprocessing.scale(
        fraglet_ds.contour.values[:,:,1],
        axis=1
    )
#     sqare_root_velocity.values[:] = np.gradient(normlized_contours.values, axis = 1)
#     sqare_root_velocity.values[:] /= np.sqrt(np.linalg.norm(sqare_root_velocity.values, axis = 2))[:,:,np.newaxis]
    
    fraglet_ds['contour_norm'] = normlized_contours
#     fraglet_ds['square_root_velocity'] = sqare_root_velocity
    
def equalize_fraglet_numbers(dataset):
    """ Equalize number of fraglets per style by selecting a random subset for over-represented styles
    """
    style_array = np.array(dataset.style.values)
    # Figure out maximum number by style
    styles, counts = np.unique(style_array, return_counts=True)
    print("Equalized fraglet number to {} per style.".format(counts[0]))
    num_per_style = np.min(counts)
    full_selection = []
    for s in styles:
        indices = np.argwhere(style_array == s).squeeze()
        selection = np.random.choice(indices, size=num_per_style, replace=False)
        full_selection.append(selection)
    full_selection = np.sort(np.concatenate(full_selection))
    return dataset.isel(fraglet_id =full_selection)

def encode_labels(dataset):
    """ Add label-encoding to dataset, label encoder will be saved as an attribute of the datset
    
    Dataset will be modified in-place
    """
    full_label = dataset['style'].astype('str').str.cat(
        dataset['allograph'].astype('str')
    ).values
    full_label_encoder = LabelEncoder()
    dataset['full_label'] = ('fraglet_id', full_label_encoder.fit_transform(full_label))
    dataset.attrs['full_label_encoder'] = full_label_encoder

    style_label = dataset['style'].astype('str').values
    style_label_encoder = LabelEncoder()
    dataset['style_label'] = ('fraglet_id', style_label_encoder.fit_transform(style_label))
    dataset.attrs['style_label_encoder'] = style_label_encoder
    
    allo_label = dataset['allograph'].astype('str').values
    allo_label_encoder = LabelEncoder()
    dataset['allo_label'] = ('fraglet_id', allo_label_encoder.fit_transform(allo_label))
    dataset.attrs['allo_label_encoder'] = allo_label_encoder

    
class CodebookStyleClassifier:
    def get_fragment_density(self, dataset, fit_feature_selection = False):
        """ Calculate density of the embedding per image
        
        Returns:
        dict of dict {'img_id' -> {'density' : ..., 'style' : ...}}
        """
        features = np.stack([dataset['contour_norm'].values], axis = 2).reshape(-1,200)
        print("Calculating fraglet embedding...")
        embedding = self.codebook_umap.transform(features)
        print("Quantizing fraglet emedding...")
        quantized_embedding = self.codebook_kmeans.predict(embedding)

        img_ids = dataset['img_id'].values.astype('str')
        
        unique_images = np.unique(img_ids)
        
        output = {}
        for i, img_id in tqdm.tqdm(enumerate(unique_images), desc="Calculating codebook pdf"):
            quantized_fragment = quantized_embedding[img_ids == img_id]
            # Count freqency of each codebook point
            output[img_id] = dict()
            density = np.bincount(
                quantized_fragment, 
                minlength=self.codebook_kmeans.n_clusters
            ).astype('float')
            # Zero values are problematic for chi2 so set a small minimum
            density = np.maximum(density, 0.001)
            # Normalize the density
            output[img_id]['img_id'] = img_id
            output[img_id]['density'] = density / np.sum(density)
            # Get an arbitrary index for this image
            first_idx = np.argmax(img_ids == img_id)
            if fit_feature_selection:
                output[img_id]['style'] = dataset['style'].values[first_idx]
        if fit_feature_selection:
            self.codebook_selection = feature_selection.SelectKBest(
                score_func = feature_selection.chi2, 
                k=self.num_codebook_vectors
            )
            # Fit codebook feature selection to maximize performance on style classification
            # of the labeled fraglets
            densities = np.array([im['density'] for im in output.values()])
            labels = np.array(
                self.style_encoder.transform(
                    [im['style'] for im in output.values()]
                )
            )
            self.codebook_selection.fit(
                densities,
                labels
            )
        for frag_dict in output.values():
            frag_dict['codebook_density'] = self.codebook_selection.transform(
                frag_dict['density'].reshape(1, -1)
            ).squeeze()
            # Normalize codebook PDF
            frag_dict['codebook_density'] /= np.sum(frag_dict['codebook_density'])
        return output
    def classify_fragment(self, query_dict, drop_self = True):
        """ Style classification using a dict as returned by get_fragment_density 
        
        """
        fragment_density = self.fragment_density
        # Drop this id, if evaluating perfomance on the same dataset
        if drop_self:
            fragment_density = {k: v for k, v in fragment_density.items() if k != query_dict['img_id']}
        # Calculate chi2 distance for each 
        distances = []
        for img_id, frag_dict in fragment_density.items():

            a = query_dict['codebook_density']
            b = frag_dict['codebook_density']
            chisq = np.sum((a - b)**2 / (a + b))
            distances.append((chisq, frag_dict['style'], img_id))
        ordering = np.argsort([dist for dist, _, _ in distances])
        distances = [distances[i] for i in ordering]
        print("Classifying {}, styles of top 3 most similar fragments: {}, {}, {}".format(
            query_dict['img_id'],
            distances[0][1], 
            distances[1][1],
            distances[2][1], 
        ))
        return distances
    def __init__(self, codebook_dataset, fragment_dataset, args):
        self.args = args
        self.preprocessing_args = pickle.loads(
            codecs.decode(
                fragment_dataset.attrs['preprocessing_args'].encode(), "base64"
            )
        )
        self.fraglet_extraction_args = pickle.loads(
            codecs.decode(
                fragment_dataset.attrs['fraglet_extraction_args'].encode(), "base64"
            )
        )
        self.allo_encoder = codebook_dataset.attrs['allo_label_encoder']
        self.style_encoder = codebook_dataset.attrs['style_label_encoder']
        self.full_encoder = codebook_dataset.attrs['full_label_encoder']
        self.codebook_umap = umap.UMAP(
            min_dist=0.1,
            n_components=args.embedding_components,
            n_neighbors=args.embedding_neighbours,
            metric=args.embedding_metric,
            verbose=True
        )
        train_features = np.stack([codebook_dataset['contour_norm'].values], axis = 2).reshape(-1,200)
        # Embed the codebook datastet
        print("Calculating codebook embedding...")
        self.codebook_embedding = self.codebook_umap.fit_transform(train_features, y=codebook_dataset.full_label)
        
        # Find codebook clusters
        print("Quantizing codebook vectors...")
        self.codebook_kmeans = cluster.KMeans(
            n_clusters = args.kmeans_clusters
        )
        self.codebook_encoding = self.codebook_kmeans.fit_predict(self.codebook_embedding)
        
        print("Embedding style fragments...")
        #fragment_features = np.stack([fragment_dataset['contour_norm'].values], axis = 2).reshape(-1,200)
        self.num_codebook_vectors = args.codebook_vectors
        self.fragment_density = self.get_fragment_density(fragment_dataset, fit_feature_selection=True)
        
        print('Evaluating training performance...')
        for img_id, frag_dict in tqdm.tqdm(self.fragment_density.items()):
            r = self.classify_fragment(frag_dict)
#         self.fragment_embedding = self.codebook_umap.transform(fragment_dataset)
        
#         print("Quantizing fragment fraglets...")
#         self.fragment_encoding = codebook_kmeans.fit_predict(self.fragment_embedding)
        
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a style classifier')
    parser.add_argument('fragment_workdir', type=check_dir, help='Work directory for style-labeled fraglets')
    parser.add_argument('codebook_workdir', type=check_dir, help='Work directory for style- and allograph-labeled fraglets')
    parser.add_argument('classifier_dir', type=make_dir, help='Directory where the classifier is saved and performance metrics are saved')
    parser.add_argument('-a', '--min_area', type=float, default=400., help='Minimum fraglet area in pixel^2')
    parser.add_argument('-p', '--min_periphery_factor', type=float, default=5, help='Minimum fraglet factor periphery / sqrt(Area) (increase to filter out blob-like fragments)')
    parser.add_argument('-c', '--embedding_components', type=int, default=2, help='Number of dimensions for the UMAP embedding')
    parser.add_argument('-n', '--embedding_neighbours', type=int, default=5, help='Number of neighbours for the UMAP embedding')
    parser.add_argument('-m', '--embedding_metric', type=str, default='manhattan', help='Distance metric for the UMAP embedding')
    parser.add_argument('-k', '--kmeans_clusters', type=int, default=300, help='Number of kmeans clusters before codebook vector selection')
    parser.add_argument('-v', '--codebook_vectors', type=int, default=50, help='Number of codebook vectors after selection')
    args = parser.parse_args()
    
    fragment_fraglets = read_fraglets(args.fragment_workdir, args)
    codebook_fraglets = read_fraglets(args.codebook_workdir, args)
    
    # Equalize number of fragments by style
    codebook_fraglets = equalize_fraglet_numbers(codebook_fraglets)
    
    normalize_fraglets(codebook_fraglets)
    normalize_fraglets(fragment_fraglets)

    
    # Encode labels of the codebook to integer labels
    encode_labels(codebook_fraglets)
    
    # Train the classifier
    classifier = CodebookStyleClassifier(codebook_fraglets, fragment_fraglets, args)
    classifier_path = os.path.join(args.classifier_dir, 'classifier.joblib')
    joblib.dump(classifier, classifier_path)
    
    #clf = joblib.load(classifier_path) 
#     # Create codebook
#     regressor = umap.UMAP()
#     codebook_data = codebook_fraglets['contour_norm'].values.reshape(-1,200)
#     fragment_fraglets = fragment_fraglets['contour_norm'].values.reshape(-1,200)
    
#     codebook_style = codebook_fraglets['style'].values.astype('str')
#     # https://umap-learn.readthedocs.io/en/latest/supervised.html?highlight=classification#using-labels-to-separate-classes-supervised-umap
    
#     label_encoder = LabelEncoder()
#     style_encoder = LabelEncoder()
#     style_encoder.fit(codebook_style)
#     label = codebook_fraglets['style'].astype('str').str.cat(codebook_fraglets['allograph'].astype('str')).values

#     target = label_encoder.fit_transform(label)
#     codebook_embedding = regressor.fit_transform(codebook_data, y=target)
    
#     fig, ax = plt.subplots(1, figsize=(14, 10))
#     plt.scatter(*codebook_embedding.T, s=0.1, c=style_encoder.transform(codebook_style), cmap='Spectral', alpha=1.0)
#     plt.setp(ax, xticks=[], yticks=[])
#     plt.title('Supervised UMAP embbedding');
#     plt.show()
    
#     fragments_embedding = regressor.transform(fragment_fraglets)
#     fig, ax = plt.subplots(1, figsize=(14, 10))
#     plt.scatter(*codebook_embedding.T, s=0.1, c=style_encoder.transform(codebook_style), alpha=1.0)
#     plt.scatter(*fragments_embedding.T, s=0.1, c='b', alpha=1.0)
#     plt.setp(ax, xticks=[], yticks=[])
#     plt.title('Supervised UMAP embbedding (codebook and fragments)');
#     plt.show()