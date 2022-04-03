import torchvision
import cv2
import os
import faiss
import heapq

import torchvision.transforms as transforms
import numpy as np
import pickle as pkl


from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FaissRetrieval(object):
    # Init Variables
    def __init__(self, dataset, train, test, train_dir, test_dir, similarity) -> None:
        # Embeddings Dimension
        self.TRAIN_DIM = train[0]
        self.TEST_DIM = test[0]
        self.EMBEDDING_DIM = train[1]

        self.TRAIN_DIR = train_dir
        self.TEST_DIR = test_dir

        # Train Variables
        self.DIC_PATH = '/faiss-web-service/resources/dictionary'
        self.INDEX_KEY = "IDMap,PCA32,IVF32,PQ16"
        self.USE_GPU = False
        self.DATASET = dataset
        self.INDEX_PATH = "/faiss-web-service/resources/index"
        self.IDS_VECTORS_PATH = '/faiss-web-service/resources/ids_paths_vectors'

        # Search Variables
        self.SIMILARITY = similarity
    

    # Load Data
    def init_dataloaders(self):
        # Dataset Transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Folders
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(self.DATASET,'train'), transform=transform)
        test_dataset = torchvision.datasets.ImageFolder(os.path.join(self.DATASET,'test'), transform=transform)

        # Paths map(list, zip(*lot))
        self.train_images_path, self.train_labels = map(list, zip(*[(path, label) for path, label in train_dataset.imgs]))
        self.test_images_path, self.test_labels = map(list, zip(*[(path, label) for path, label in test_dataset.imgs]))

        # Load Images
        self.train_images = np.array([cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in self.train_images_path])
        self.test_images = np.array([cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in self.test_images_path])

        # Images shape
        print(f"Train Images : {self.train_images.shape}")
        print(f"Test Images : {self.test_images.shape}")

    # Retrieve Embeddings
    def get_embeddings(self, split):
        if split == "train":
            dim = (self.TRAIN_DIM, self.EMBEDDING_DIM)
            path = self.TRAIN_DIR
        else:
            dim = (self.TEST_DIM, self.EMBEDDING_DIM)
            path = self.TEST_DIR

        a = pkl.load(open(path, "rb"))
        a = np.array([item for sublist in a for item in sublist]).reshape((1881, 10))
        embedding_features = np.zeros((1881, 65))
        embedding_features[:a.shape[0],:a.shape[1]] = a

        # Normalize
        normalizer = StandardScaler() # Normalize Data
        embedding_features = normalizer.fit_transform(embedding_features)
        pca = PCA(n_components=64) # Keep 64 var
        embedding_features = pca.fit_transform (embedding_features)
        self.EMBEDDING_DIM = embedding_features.shape[1]


        return embedding_features
    
    # Train FAISS
    def train(self):
        # Read Data
        print("Reading Data...")
        self.init_dataloaders()

        # Get Emebeddings
        print("Getting Embeddings...")
        features = self.get_embeddings(split = "train")

        # Init FAISS
        self.index = faiss.index_factory(self.EMBEDDING_DIM, self.INDEX_KEY)

        # Send to GPU
        if self.USE_GPU:
            print("Use GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # Train
        print('Starting Training...')
        ids_count = 0.
        self.index_dict = {}
        ids = []


        for idx, file_name in enumerate(self.train_images_path):
            image_dict = {ids_count: (file_name, features[idx])}
            self.index_dict.update(image_dict)
            ids_count += 1
            ids.append(idx)
            
            if ids_count % 9000000 == 8999999:
                if not self.index.is_trained and self.INDEX_KEY != "IDMap,Flat":
                    self.index.train(features)
                    self.index.add_with_ids(features, ids)
                    ids = None
        ids = np.array(ids)
        if ids.any():
            if not self.index.is_trained and self.INDEX_KEY != "IDMap,Flat":
                self.index.train(np.ascontiguousarray(features).astype(np.float32))
                print('Training: DONE')
                self.index.add_with_ids(np.ascontiguousarray(features).astype(np.float32), np.ascontiguousarray(ids).astype(np.int64))
                print('Adding IDs: DONE')

        
    
    def search_by_image(self, feature, k):
        ids = [None]
        return self.__search__(ids, [feature], k)
    

    def __search__(self, ids, vectors, k):
        
        results = []

        for id_, feature in zip(ids, vectors):
            scores, neighbors = self.index.search(feature.astype(np.float32), k = k) if feature.size > 0 else ([], [])
            r = {"scores": scores, "neighbors": neighbors}
            results.append(r)
            
        return results

        








        
