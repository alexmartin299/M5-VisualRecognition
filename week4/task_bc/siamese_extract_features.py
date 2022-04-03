from networks import EmbeddingNet, SiameseNet
from losses import ContrastiveLoss
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit
import numpy as np
import torchvision.transforms as transforms
import torchvision
import os
cuda = torch.cuda.is_available()
from datasets import SiameseNetworkDataset
import matplotlib.pyplot as plt

BATCH_SIZE = 8
DATASET = '../MIT_split'
#load data and transform to siamese format
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = torchvision.datasets.ImageFolder(os.path.join(DATASET,'train'))
test_dataset = torchvision.datasets.ImageFolder(os.path.join(DATASET,'test'))


siamese_train_dataset = SiameseNetworkDataset(train_dataset,transform) # Returns pairs of images and target same/different
siamese_test_dataset = SiameseNetworkDataset(test_dataset,transform)

siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

embedding_net = EmbeddingNet()

model = SiameseNet(embedding_net)
if cuda:
    model.cuda()

margin = 1.
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 500

fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
