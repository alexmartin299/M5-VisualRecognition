import os
import cv2
import torch
import torchvision


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import torchvision.transforms as transforms

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, label_binarize
from cf_matrix import *
from faiss_custom import FaissRetrieval

DATASET = '../../MIT_split'


faiss = FaissRetrieval(dataset = DATASET, train = (1881, 2048), test = (807, 2048),
                       train_dir = "../features/train_features_resnet50.pkl", 
                       test_dir = "../features/test_features_resnet50.pkl", similarity = 5)

faiss.train()

embedding_features = pkl.load(open("../features/train_features_resnet50.pkl", "rb"))
embedding_features = np.array([item for sublist in embedding_features for item in sublist]).reshape((1881, 2048))
normalizer = StandardScaler() # Normalize Data
embedding_features = normalizer.fit_transform(embedding_features)
pca = PCA(n_components=64) # Keep 64 var
embedding_features = pca.fit_transform (embedding_features)
results = faiss.search_by_image(feature=embedding_features[10].reshape((1, embedding_features.shape[1])), k = 3)
print(results)