import numpy as np
import os
import scipy.io

from utils import select_img_from_feats, select_text_from_feats, generate_negative_idx, aggregate_text_features, reduce_features_img, compute_distance

# Load features from db
path_to_db = '../flickr30k_images'

# Load features from db
features_text = np.load(os.path.join(path_to_db, 'fasttext_feats.npy'), allow_pickle=True)
features_img = scipy.io.loadmat(os.path.join(path_to_db, 'vgg_feats.mat'))['feats']

print(np.transpose(features_img).shape)
features_img_reduced = reduce_features_img(np.transpose(features_img))
print(features_img_reduced.shape)
# select randomly an image
idx_img, anchor = select_img_from_feats(features_img_reduced)

# select randomly a caption from the image
idx_img, _, feature_text = select_text_from_feats(features_text, idx_img=idx_img)

# select randomly a caption from other image (cannot be the anchor)
idx_img_neg, _, feature_text_neg = select_text_from_feats(features_text,
                                                       idx_img=generate_negative_idx(features_img.shape[0], idx_img))
#feature_img = feature_img.reshape(1, -1)
print(anchor.shape, feature_text.shape, feature_text_neg.shape)

# project the features into the same space
positive, negative = aggregate_text_features(feature_text, feature_text_neg, mode='mean')

print(anchor.shape, positive.shape, negative.shape)

dist2positive, dist2negative = compute_distance(anchor,positive,negative)

print(dist2positive, dist2negative)

print('finished')
