import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import normalize
from scipy.spatial.distance import euclidean, chebyshev, minkowski

def compute_similiarity(anchor, positive, negative, metric='euclidean'):

    if metric=='euclidean':
        return euclidean(anchor,positive), euclidean(anchor,negative)
    elif metric=='chebyshev':
        return chebyshev(anchor,positive), chebyshev(anchor,negative)
    elif metric=='minkowski':
        return minkowski(anchor,positive), minkowski(anchor,negative)
    elif metric=='dot_product':
        return np.dot(anchor,positive), np.dot(anchor, negative)
    else:
        print('Non-valid metric')

def predicted_label(anch2pos_sim, anch2neg_sim, mode):
    if mode in ['euclidean','chebyshev','minkowski','dot-product']:
        return int(anch2pos_sim<anch2neg_sim)


def select_img_from_feats(features_img, idx_img=None):
    """
    Selects the features of the images with the given ids. If the id is not given, it returns a random image.
    :param features_img: The features of the images, e.g. vgg features.
    :param idx_img: The ids of the image to select.
    :return: the features of the selected image.
    """

    if idx_img is None:
        idx_img = np.random.randint(0, features_img.shape[0])

    assert 0 <= idx_img < features_img.shape[0], "The id of the image is out of range."

    return idx_img, features_img[idx_img,: ]


def select_text_from_feats(features_text, idx_img=None, idx_text=None):
    """
    Selects the features of the texts with the given ids. If the id is not given, it returns a random text.
    :param features_text: The features of the texts, e.g. vgg features.
    :param idx_img: The ids of the image to select.
    :param idx_text: The ids of the text to select.
    :return: the features of the selected text.
    """

    if idx_text is None:
        idx_text = np.random.randint(0, features_text.shape[1])

    assert 0 <= idx_text < features_text.shape[1], "The id of the text is out of range."

    if idx_img is None:
        idx_img = np.random.randint(0, features_text.shape[0])

    assert 0 <= idx_img < features_text.shape[0], "The id of the image is out of range."

    return idx_img, idx_text, features_text[idx_img, idx_text]


def aggregate_text_features(positive_caption, negative_caption, mode):
    """
    Aggregates the positive and negative caption.
    :param anchor: The anchor image.
    :param positive_caption: The positive caption.
    :param negative_caption: The negative caption.
    :param mode: method for aggregating the text embedding
    :return: The projected anchor and the positive and negative caption.
    """
    if mode == 'mean':
        norm_positive = normalize(np.mean(positive_caption, axis=0)[np.newaxis,:], axis=1)
        norm_negative = normalize(np.mean(negative_caption, axis=0)[np.newaxis,:], axis=1)
        return np.squeeze(norm_positive), np.squeeze(norm_negative)
    if mode == 'sum':
        norm_positive = normalize(np.sum(positive_caption,axis=0)[np.newaxis,:], axis=1)
        norm_negative = normalize(np.sum(negative_caption,axis=0)[np.newaxis,:], axis=1)
        return np.squeeze(norm_positive), np.squeeze(norm_negative)
    if mode == 'PCA':
        """Applies PCA"""
        norm_positive = normalize(np.transpose(positive_caption), axis=1)
        norm_negative = normalize(np.transpose(negative_caption), axis=1)
        pca = PCA(n_components=1)
        return pca.fit_transform(norm_positive), pca.fit_transform(norm_negative)
    if mode == 'ICA':
        """Applies ICA"""
        norm_positive = normalize(np.transpose(positive_caption), axis=1)
        norm_negative = normalize(np.transpose(negative_caption), axis=1)
        ica = FastICA(n_components=1)
        return ica.fit_transform(norm_positive), ica.fit_transform(norm_negative)

def reduce_features_img(features_img, method='PCA'):
    """
    :param features_img:
    :return: vectors reduced to 300 features
    """
    norm_features = normalize(features_img, axis=1)
    if method == 'PCA':
        pca = PCA(n_components=300)
        return pca.fit_transform(norm_features)
    if method == 'ICA':
        ica = FastICA(n_components=300)
        return ica.fit_transform(norm_features)

def generate_negative_idx(range, exception_idx):
    """
    Returns a random integer between 0 and range. This number can't be exception_idx
    :param range: The range of the random integer.
    :param exception_idx: The exception idx.
    :return: The random integer.
    """
    while True:
        idx = np.random.randint(0, range)
        if idx != exception_idx:
            return idx