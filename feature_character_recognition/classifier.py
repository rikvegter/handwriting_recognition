import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle

class RandomForestClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL):
