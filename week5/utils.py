import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

def compute_distance(anchor, positive, negative, metric='euclidean'):

    if metric=='euclidean':
        return euclidean(anchor,positive), euclidean(anchor,negative)

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
    Projects the anchor and the positive and negative caption to the same dimension using PCA.
    :param anchor: The anchor image.
    :param positive_caption: The positive caption.
    :param negative_caption: The negative caption.
    :return: The projected anchor and the positive and negative caption.
    """
    if mode == 'mean':
        return np.mean(positive_caption,axis=0), np.mean(negative_caption,axis=0)


def reduce_features_img(features_img):
    pca = PCA(n_components=300)
    return pca.fit_transform(features_img)


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