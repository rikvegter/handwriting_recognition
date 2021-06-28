import xarray as xr
import numpy as np
import argparse
import os
import tqdm
from train_classifier import CodebookStyleClassifier, check_dir, read_fraglets, normalize_fraglets
from preprocess_images import get_preprocessed_images_dataset, make_dir
from extract_fraglets import get_fraglet_dataset
import joblib

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Classify fragment style based on labeled fraglets and fragment fraglets')
    parser.add_argument('image_dir', type=check_dir, help='Image directory')
    parser.add_argument('classifier_dir', type=check_dir, help='Work directory containing the saved classifier')
    parser.add_argument('results_dir', type=make_dir, help='Directory for writing classification results')
    args = parser.parse_args()
    
    # Load classifier
    classifier_path = os.path.join(args.classifier_dir, 'classifier.joblib')
    classifier = joblib.load(classifier_path)
    
    # Preprocess fragments
    preprocessing_args = classifier.preprocessing_args
    preprocessing_args.image_dir = args.image_dir
    preprocessing_args.style = False
    preprocessing_args.labeled = False
    preprocessing_args.pattern = "*.*"
    preprocessing_args.name = "style_classification"
    
    image_data = get_preprocessed_images_dataset(preprocessing_args)
    fraglet_extraction_args = classifier.fraglet_extraction_args
    fraglet_extraction_args.augment_times = 3
    classification_fraglet_data = get_fraglet_dataset(image_data, fraglet_extraction_args)
    dataset_path = os.path.join(args.classifier_dir, 'fraglets.nc')
    
#     print('Writing output to {}'.format(dataset_path))
#     classification_fraglet_data.to_netcdf(dataset_path, encoding = {
#         "contour": {"zlib" : True},
#     })
    
    normalize_fraglets(classification_fraglet_data)
    fragment_density = classifier.get_fragment_density(classification_fraglet_data)
    result_path = make_dir("./results")
    for img_id, frag_dict in fragment_density.items():
        r = classifier.classify_fragment(frag_dict)
        path = image_data.where(image_data.img_id == img_id, drop=True).img_path.item()
        basename = os.path.splitext(os.path.basename(path))[0]
        style_path = os.path.join('./results', basename + "_style.txt")
        print(style_path)
        with open(style_path, 'w') as style_out_file:
            style_out_file.write(r[0][1])

#     # Load fragments
#     fragment_fraglets = read_fraglets(args.fragment_dir, classifier.args)
#     normalize_fraglets(fragment_fraglets)
                        
#     fragment_density = classifier.get_fragment_density(fragment_fraglets)
                        
#     for img_id, frag_dict in fragment_density.items():
#         r = classifier.classify_fragment(frag_dict)