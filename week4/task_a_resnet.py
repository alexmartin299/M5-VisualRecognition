import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle as pkl
import torchvision
import torchvision.transforms as transforms
import os
import copy


from tqdm import tqdm
from torchvision.models import resnet50

# Initialization
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)


# Define Parameters
DATASET = '../MIT_split'
BATCH_SIZE = 8
NUM_CLASS = 8
EPOCHS = 5

# Model
model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASS)
model.to(device)
params_to_update = model.parameters()

OPT = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
LOSS = nn.CrossEntropyLoss()



# Load Dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(os.path.join(DATASET,'train'), transform=transform)
test_dataset = torchvision.datasets.ImageFolder(os.path.join(DATASET,'test'), transform=transform)

train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
dataloaders = {"train": train_dataloader, "test": test_dataloader}


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    # Keep Best
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # Track Accuracy
    val_acc_history = []

    # Train
    for epoch in tqdm(range(num_epochs), desc="Train Resnet50..."):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        

        # Train and Evaluate
        for phase in ["train", "test"]:
            # Set Modes
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            # Initialize Metrics
            running_loss = 0.0
            running_corrects = 0

            # Iterate Data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Extract output for batch
                    outputs = model(inputs)
                    # Evaluate Loss
                    loss = criterion(outputs, labels)
                    # Make Predictions
                    _, preds = torch.max(outputs, 1)
                    # Backward + optimize in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # Update Metrics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Epoch Info
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            
        print()

    # Load Best Stats
    model.load_state_dict(best_model_wts)
    # Save
    torch.save(model.state_dict(), "fine_tuned_resnet50.pt")
    return model, val_acc_history


def extract_features(model, x, phase):
    ### strip the last layer
    outputs = []
    with torch.no_grad():
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        for inputs, _ in tqdm(x[phase], desc = f"Extracting {phase} Features..."):
            inputs = inputs.to(device)
            output = feature_extractor(inputs)
            outputs.append(output.cpu().detach().numpy())
        print(outputs[0].shape)
        return outputs


if not os.path.exists("./fine_tuned_resnet50.pt"):
    best_model, _ = train_model(model=model, dataloaders = dataloaders, criterion = LOSS, optimizer = OPT, num_epochs=EPOCHS)
else:
    best_model = copy.deepcopy(model)
    best_model.load_state_dict(torch.load("fine_tuned_resnet50.pt"))
    
best_model.eval()
train_features = extract_features(best_model,dataloaders,"train")
test_features = extract_features(best_model,dataloaders,"test")

pkl.dump(train_features, open("train_features_resnet50.pkl", "wb"))
pkl.dump(test_features, open("test_features_resnet50.pkl","wb"))



