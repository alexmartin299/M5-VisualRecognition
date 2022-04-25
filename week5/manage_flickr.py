import json

import numpy as np
import pickle

PATH_TO_DATASET = '../flickr30k_images'

# Read FastText Features in .pkl format
with open('./features/image_features_fastrcnn.pkl', 'rb') as f:
    fastrcnn_feats = pickle.load(f)

fastrcnn_feats = np.squeeze(np.asarray(fastrcnn_feats)).T

# Read the JSON files of image captions
with open(PATH_TO_DATASET + '/train.json', 'rb') as f:
    train_ann = json.load(f)
with open(PATH_TO_DATASET + '/test.json', 'rb') as f:
    test_ann = json.load(f)
with open(PATH_TO_DATASET + '/val.json', 'rb') as f:
    val_ann = json.load(f)

train_ids = [ann['imgid'] for ann in train_ann]
test_ids = [ann['imgid'] for ann in test_ann]
val_ids = [ann['imgid'] for ann in val_ann]

# Train features form list if ids
train_feats = fastrcnn_feats[:,train_ids]
test_feats = fastrcnn_feats[:,test_ids]
val_feats = fastrcnn_feats[:,val_ids]

# Store the features in .pkl format
with open(PATH_TO_DATASET + '/train_fasterrcnn_features.pkl', 'wb') as f:
    pickle.dump(train_feats, f)
with open(PATH_TO_DATASET + '/test_fasterrcnn_features.pkl', 'wb') as f:
    pickle.dump(test_feats, f)
with open(PATH_TO_DATASET + '/val_fasterrcnn_features.pkl', 'wb') as f:
    pickle.dump(val_feats, f)

print(
    "Faster RCNN Features: {}\nTrain Annotations: {}\nTest Annotations: {}\nVal Annotations: {}".format(
        fastrcnn_feats.shape, len(train_ann), len(test_ann), len(val_ann)))