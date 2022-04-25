import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle as pkl
import torchvision
import torchvision.transforms as transforms
import os
import copy

from torchsummary import summary
from tqdm import tqdm
from torchvision.models import mobilenet_v3_large, resnet50
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
# Imports
import torch, json
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image

# Import matplotlib and configure it for pretty inline plots
import matplotlib.pyplot as plt

# Choose an image to pass through the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATASET = '../flickr30k_images'
BATCH_SIZE = 8
# Prepare the labels
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
# First prepare the transformations: resize the image to what the model was trained on and convert it to a tensor
# Load the image
train_dataset = torchvision.datasets.ImageFolder(os.path.join(DATASET,'train'), transform=transform)

train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
# Now apply the transformation, expand the batch dimension, and send the image to the GPU

dataloaders = {"1": train_dataloader}

def extract_features(model, x, phase):
    ### strip the last layer
    outputs = []
    with torch.no_grad():
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        for inputs, _ in tqdm(x[phase], desc = f"Extracting {phase} Features..."):
            inputs = inputs.to(device)
            output = feature_extractor(inputs)
            outputs.extend(output.cpu().detach().numpy())
        print(outputs[0].shape)
        return outputs

model = mobilenet_v3_large(pretrained=True)
model.to(device)
model.eval()
train_features = extract_features(model,dataloaders,"1")

pkl.dump(train_features, open("./features/image_features_fastrcnn.pkl", "wb"))




