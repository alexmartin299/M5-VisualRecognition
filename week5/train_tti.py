import os
import os.path
from os import path

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataloader import Flickr30k, TripletFlickr30kText2Img

from trainer import fit
from models_utils import EmbeddingImageNN, EmbeddingTextNN, TripletText2Image, TripletLoss

def main(out_size=4096, text_aggregation='sum'):
    # cuda management
    cuda = torch.cuda.is_available()
    # Find which device is used

    # Output directory
    OUTPUT_MODEL_DIR = './models/'

    # Create the output directory if it does not exist
    if not path.exists(OUTPUT_MODEL_DIR):
        os.makedirs(OUTPUT_MODEL_DIR)

    # Load the datasets
    TRAIN_IMG_EMB = "../flickr30k_images/train_fasterrcnn_features.pkl"
    TEST_IMG_EMB = "../flickr30k_images/val_fasterrcnn_features.pkl"
    TRAIN_TEXT_EMB = "../flickr30k_images/train_fasttext_features.pkl"
    TEST_TEXT_EMB = "../flickr30k_images/val_fasttext_features.pkl"

    # Method selection
    base = 'tti'
    image_features = 'FasterRCNN'
    emb_size = 300
    info = 'out_size_' + str(out_size)
    model_id = base + '_' + image_features + '_' + text_aggregation + '_textagg_' + info

    # Load the datasets
    train_dataset = Flickr30k(TRAIN_IMG_EMB, TRAIN_TEXT_EMB, train=True,
                              text_aggregation=text_aggregation)  # Create the 1 dataset
    test_dataset = Flickr30k(TEST_IMG_EMB, TEST_TEXT_EMB, train=False,
                             text_aggregation=text_aggregation)  # Create the test dataset

    train_dataset_triplet = TripletFlickr30kText2Img(train_dataset, split='1')
    test_dataset_triplet = TripletFlickr30kText2Img(test_dataset, split='test')

    batch_size = 1024

    # Create the dataloaders
    triplet_train_loader = torch.utils.data.DataLoader(train_dataset_triplet, batch_size=batch_size, shuffle=True)
    triplet_test_loader = torch.utils.data.DataLoader(test_dataset_triplet, batch_size=batch_size, shuffle=False)

    margin = 1.
    embedding_text_net = EmbeddingTextNN(embedding_size=emb_size, output_size=out_size)
    embedding_image_net = EmbeddingImageNN(output_size=out_size)
    model = TripletText2Image(embedding_text_net, embedding_image_net, margin=margin)

    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    start_epoch = 0
    # Check if file exists
    if path.exists(OUTPUT_MODEL_DIR + model_id + '.pth'):
        print('Loading the model from the disk')
        checkpoint = torch.load(OUTPUT_MODEL_DIR + model_id + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    print('Starting training, EPOCH: ', start_epoch)
    n_epochs = 26
    log_interval = 10

    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
        model_id, start_epoch=start_epoch)

sizes = [256, 512, 1024, 2048, 4096]
for method in ['sum', 'mean']:
    for size in sizes:
        main(size, method)


