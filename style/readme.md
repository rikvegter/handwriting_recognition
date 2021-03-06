# Pipeline
Style classification consists of four steps:

1. Load images from file system and save preprocesed images as a labeled dataset
2. Extract contours from the image dataset, augment the data and split it into fraglets
3. Train a style classifier based on (1) a set of extracted graphemes labeled with both style and allograph and (2) a set of fragments labeled by style
      - Fraglets are filtered based on perimeter-to-area ratio and fraglet area to remove uninformative and small fraglets
      - Fraglets are normalized
      - A random sample of the fraglets is selected such that the number of fraglets per style is equal
      - The fraglets are used to learn a supervised UMAP embedding that separates based on allographs and styles
      - K-means is used to find clusters in the embedded fraglets that will be used as a codebook
      - The labeled fragments are encoded by assigning each fraglet to the most similar codebook fraglet
      - A Chi-squared based method is used to select the most informative codebook vectors for style classification
      - The PDF of the style-labeled fragments is computed
      - The classifier is saved (which consists of the embedding, the codebook means, the selected codebook vectors and the fragment pdfs)
4. Label new fragments using a saved classifier
      - Preprocess the images and extract the fraglets
      - Filter and normalize fraglets
      - Compute codebook PDF of the fraglets
      - Find most similar fragment(s)
      - If requested, write out the style of the best match to a text file in the image directory for each classified fragment

# Scripts

## `classify_style.py`

Should be called with three arguments:
 - Input image directory
 - Classifier directory
 - Result directory

e.g.:

```bash
python3 classify_style.py /patyh
```

Calculating the UMAP embedding of the fraglets may take a while

The classifier is a pickled sklearn/umap model and may fail to load due to incompatible libary versions, in this case the classifier should be re-trained. 

# Dependencies

 - `numpy`, `scipy` and `matplotlib` for basic scientific functions
 - `scikit-image` for image processing
 - `xarray` and `xarray-dataclasses` for managing the datasets
 - `scikit-learn` for feature selection and classification
 - `umap-learn` for metric learning/dimension reduction
 - `tqdm` for showing progress bars
 