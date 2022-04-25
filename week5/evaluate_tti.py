import json
import os.path
import pickle
from os import path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from dataloader import Flickr30k
from models_utils import EmbeddingImageNN, EmbeddingTextNN, TripletText2Image
from metrics import mapk

cuda = torch.cuda.is_available()


def extract_embeddings(dataloader, model, out_size=256, model_id=''):
    model.to('cuda')
    with torch.no_grad():
        model.eval()
        image_embeddings = np.zeros((len(dataloader.dataset), out_size))
        text_embeddings = np.zeros((len(dataloader.dataset) * 5, out_size))
        k = 0
        for images, texts in dataloader:
            if cuda:
                images = images.cuda()
                texts = texts.cuda()

            im_emb, text_emb = model.get_embedding_pair(images, texts)
            image_embeddings[k:k + len(images)] = im_emb.data.cpu().numpy()
            text_embeddings[k:k + len(texts) * 5] = text_emb.data.cpu().numpy().reshape(len(texts) * 5, out_size)
            k += len(images)

    return image_embeddings, text_embeddings


def main_tti():
    # Load the datasets
    TEST_IMG_EMB = "../flickr30k_images/test_fasterrcnn_features.pkl"
    TEST_TEXT_EMB = "../flickr30k_images/test_fasttext_features.pkl"

    # Method selection
    base = 'tti'
    text_aggregation = 'mean'
    image_features = 'FasterRCNN'
    out_size = 512
    info = 'out_size_' + str(out_size)
    model_id = base + '_' + image_features + '_' + text_aggregation + '_textagg_' + info

    PATH_MODEL = 'models/'
    PATH_RESULTS = 'results/'
    # Create folder if it does not exist
    if not path.exists(PATH_RESULTS):
        os.makedirs(PATH_RESULTS)

    # Load the test dataset
    test_dataset = Flickr30k(TEST_IMG_EMB, TEST_TEXT_EMB, train=False,
                             text_aggregation=text_aggregation)  # Create the test dataset

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=1)

    margin = 1.
    embedding_text_net = EmbeddingTextNN(embedding_size=300, output_size=out_size, late_fusion=None)
    embedding_image_net = EmbeddingImageNN(output_size=out_size)
    model = TripletText2Image(embedding_text_net, embedding_image_net, margin=margin)

    # Check if file exists
    if path.exists(PATH_MODEL + model_id + '.pth'):
        print('Loading the model from the disk, {}'.format(model_id + '.pth'))
        checkpoint = torch.load(PATH_MODEL + model_id + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

    # Obtain ground truth from the json file (test.json)
    with open('../flickr30k_images/test.json') as f:
        data = json.load(f)

    gt = {}  # Ground truth as a dictionary with the image filename as key and the list of text id as value
    dict_sentences = {}  # Dictionary with the text id as key and the sentence as value
    count = 0
    for item in data:
        gt[item['filename']] = [x['raw'] for x in item['sentences']]
        for sentence in item['sentences']:
            dict_sentences[count] = sentence['raw']
            count += 1

    # Extract embeddings
    image_embeddings, text_embeddings = extract_embeddings(test_loader, model, out_size, model_id)
    # Compute the labels for each embedding
    image_labels = [i for i in range(1, 1000 + 1)]
    text_labels = [j for j in range(1, 1000 + 1) for i in range(5)]  # Trick to obtain the same
    # number of labels, copying the same labels 5 (5 text embeddings)

    # Compute the nearest neighbors
    print('Computing the nearest neighbors...')
    k = 1  # Number of nearest neighbors

    # # load results if exists
    # if path.exists(PATH_RESULTS + model_id + '_knn.pkl'):
    #     print('Loading the nearest neighbors from the disk, {}'.format(model_id + '_knn.pkl'))
    #     distances, indices = pickle.load(open(PATH_RESULTS + model_id + '_knn.pkl', 'rb'))
    # else:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree').fit(image_embeddings, image_labels)

    # Make predictions
    distances, indices = knn.kneighbors(text_embeddings)

    # Compute mAPk
    image_labels_pred = []

    #
    for k_predictions in indices.tolist():
        # map indices with the corresponding labels
        k_labels_pred = [image_labels[i] for i in k_predictions]
        image_labels_pred.append(k_labels_pred)

    t_labels = [[i] for i in text_labels] # Convert list of labels into list of list (for mapk function)
    map_k = mapk(t_labels, image_labels_pred, k=k)
    print(f'mAP@{k}: {map_k}')

    # Qualitative results
    num_samples = 10
    # Create random samples
    random_samples = np.random.choice(list(range(5000)), num_samples, replace=False)

    # im_labels, image_labels_pred
    for sample in random_samples:
        print("Example:" + str(sample))
        print("--------------------------------")

        print("Query text: " + dict_sentences[sample])

        # Obtain ground truth image, mapping the text sample to the image id
        gt_image_id = text_labels[sample]
        # Map the id to the image filename
        gt_image_filename = list(gt)[gt_image_id - 1]
        plt.figure(figsize=(20, 10))
        # Plot the ground truth image
        plt.subplot(1, k+1, 1)
        plt.imshow(plt.imread('../flickr30k_images/train/1/' + gt_image_filename))

        # Get predicted images from that text
        predictions = indices[sample]
        count = 1
        for pred in predictions:
            filename = list(gt)[pred]
            plt.subplot(1, k+1, count + 1)
            plt.imshow(plt.imread('../flickr30k_images/train/1/' + filename))
            count += 1
        plt.show()
        print("--------------------------------------------------------------------------------")

main_tti()