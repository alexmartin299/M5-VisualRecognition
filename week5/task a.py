import numpy as np
import os
import scipy.io
from tqdm import tqdm
from utils import select_img_from_feats, select_text_from_feats, generate_negative_idx, aggregate_text_features, reduce_features_img, compute_similiarity, predicted_label
from sklearn.metrics import average_precision_score


#este approach es de prueba no funciona obviamente pero se puede utilizar para comparar con el metric learning que se hace despues

# Load features from db
path_to_db = '../flickr30k_images'

# Load features from db
features_text = np.load(os.path.join(path_to_db, 'fasttext_feats.npy'), allow_pickle=True)
features_img = scipy.io.loadmat(os.path.join(path_to_db, 'vgg_feats.mat'))['feats']


features_img_reduced = reduce_features_img(np.transpose(features_img),method = 'PCA')
y_predicted =[]
y_real = np.zeros(155070)
for idx in tqdm(range(0,31014),desc='Images processed'):
    # select one image
    idx_img, anchor = select_img_from_feats(features_img_reduced, idx_img=idx)

    for text_idx in range(0,5):
        idx_img, _, feature_text = select_text_from_feats(features_text, idx_img=idx_img, idx_text=text_idx)
        # select randomly a caption from other image (cannot be the anchor)
        idx_img_neg, _, feature_text_neg = select_text_from_feats(features_text,
                                                               idx_img=generate_negative_idx(features_img.shape[0], idx_img))
        # project the features into the same space
        positive, negative = aggregate_text_features(feature_text, feature_text_neg, mode='sum')

        sim2positive, sim2negative = compute_similiarity(anchor, positive, negative, metric='dot_product')

        y_pred = predicted_label(sim2positive, sim2negative, mode='euclidean')

        y_predicted.append(y_pred)


ap = average_precision_score(y_predicted, y_real)
print("Average precision with whole dataset: {}".format(ap))

#este approach es de prueba no funciona obviamente pero se puede utilizar para comparar con el metric learning que se hace despues




