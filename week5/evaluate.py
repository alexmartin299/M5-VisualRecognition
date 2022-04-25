import json
import os.path
from os import path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from dataloader import Flickr30k
from models_utils import EmbeddingImageNN, EmbeddingTextNN, TripletImage2Text, TripletText2Image
from tqdm import tqdm

cuda = torch.cuda.is_available()


def extract_embeddings(dataloader, model, out_size=256, model_id=''):
    model.to('cpu')
    with torch.no_grad():
        model.eval()
        image_embeddings = np.zeros((len(dataloader.dataset), out_size))
        text_embeddings = np.zeros((len(dataloader.dataset) * 5, out_size))
        k = 0
        for images, texts in tqdm(dataloader,desc='Extracting embedding',total=1000) :
            im_emb, text_emb = model.get_embedding_pair(images, texts)
            image_embeddings[k:k + len(images)] = im_emb.data.cpu().numpy()
            text_embeddings[k:k + len(texts) * 5] = text_emb.data.cpu().numpy().reshape(len(texts) * 5, out_size)
            k += len(images)

    return image_embeddings, text_embeddings

def main_tti(text_aggregation, image_features, out_size):
    # Load the datasets
    TEST_IMG_EMB = "../flickr30k_images/test_vgg_features.pkl"
    TEST_TEXT_EMB = "../flickr30k_images/test_fasttext_features.pkl"

    # Method selection
    base = 'tti'
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
    embedding_text_net = EmbeddingTextNN(embedding_size=300, output_size=out_size)
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

    gt = {}
    dict_sentences = {}
    count = 0
    for item in data:
        gt[item['filename']] = [x['raw'] for x in item['sentences']]
        for sentence in item['sentences']:
            dict_sentences[count] = sentence['raw']
            count += 1


    image_embeddings, text_embeddings = extract_embeddings(test_loader, model, out_size, model_id)

    image_labels = [i for i in range(1, 1000 + 1)]
    text_labels = [j for j in range(1, 1000 + 1) for i in range(5)]

    k = 10
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree').fit(image_embeddings, image_labels)
    knn = knn.fit(text_embeddings, text_labels)
    distances, indices = knn.kneighbors(text_embeddings)

    knn_accuracy = knn.score(text_embeddings, text_labels)
    print('KNN accuracy: {}%'.format(100 * knn_accuracy))

    random_samples = np.arange(10, 13)
    for sample in random_samples:
        print("Example:" + str(sample))
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
            print(pred)
            filename = list(gt)[pred]
            plt.subplot(1, k+1, count + 1)
            plt.imshow(plt.imread('../flickr30k_images/train/1/' + filename))
            count += 1
        plt.show()

def main_itt(text_aggregation, image_features, out_size ):
    # Load the datasets
    TEST_IMG_EMB = "../flickr30k_images/test_vgg_features.pkl"
    TEST_TEXT_EMB = "../flickr30k_images/test_fasttext_features.pkl"

    # Method selection
    base = 'itt'
    info = 'out_size_' + str(out_size)
    model_id = base + '_' + image_features + '_' + text_aggregation + '_textagg_' + info
    print('Model_id used: {}'.format((model_id)))
    PATH_MODEL = 'models/'
    PATH_RESULTS = 'results/'
    # Create folder if it does not exist
    if not path.exists(PATH_RESULTS):
        os.makedirs(PATH_RESULTS)


    test_dataset = Flickr30k(TEST_IMG_EMB, TEST_TEXT_EMB, train=False,
                             text_aggregation=text_aggregation)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=1)

    margin = 1.
    embedding_text_net = EmbeddingTextNN(embedding_size=300, output_size=out_size)
    embedding_image_net = EmbeddingImageNN(output_size=out_size)
    model = TripletImage2Text(embedding_text_net, embedding_image_net, margin=margin)

    if path.exists(PATH_MODEL + model_id + '.pth'):
        device = torch.device('cpu')
        print('Loading the model from the disk, {}'.format(model_id + '.pth'))
        checkpoint = torch.load(PATH_MODEL + model_id + '.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    with open('../flickr30k_images/test.json') as f:
        data = json.load(f)

    gt = {}
    dict_sentences = {}
    count = 0
    for item in data:
        gt[item['filename']] = [x['raw'] for x in item['sentences']]
        for sentence in item['sentences']:
            dict_sentences[count] = sentence['raw']
            count += 1


    image_embeddings, text_embeddings = extract_embeddings(test_loader, model, out_size, model_id)

    # Compute labels
    image_labels = [i for i in range(1, 1000 + 1)]
    text_labels = [j for j in range(1, 1000 + 1) for i in range(5)]

    k = 10
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree').fit(text_embeddings, text_labels)
    knn = knn.fit(image_embeddings, image_labels)
    # Make predictions
    image_labels_pred = knn.predict(image_embeddings)
    distances, indices = knn.kneighbors(image_embeddings)

    # Compute the accuracy
    knn_accuracy = knn.score(image_embeddings, image_labels)
    print('KNN accuracy: {}%'.format(100*knn_accuracy))


    random_samples = np.arange(10,13)
    for sample in random_samples:
        print("Example:" + str(sample))

        # Get image embedding from batch
        filename = list(gt)[sample]
        print("Ground truth: ")
        for t in gt[filename]:
            print(t)

        predictions = indices[sample]
        print("Predictions:")
        for pred in predictions:
            print(dict_sentences[pred])

        im = plt.imread('../flickr30k_images/train/1/' + filename)
        plt.imshow(im)
        plt.show()
# Main
for i in os.listdir('./models/'):
    args = i.split('_')
    base = args[0]
    text_aggregation = args[2]
    image_features = args[1]
    out_size = int(args[6].split('.')[0])
    if base =='itt' and out_size==500 and image_features=='VGG':
        main_itt(text_aggregation, image_features, out_size)


