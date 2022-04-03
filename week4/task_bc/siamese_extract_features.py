from networks import EmbeddingNet, SiameseNet
from losses import ContrastiveLoss
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit
import torchvision.transforms as transforms
import torchvision
import os
cuda = torch.cuda.is_available()
from datasets import SiameseNetworkDataset
from tqdm import tqdm
import pickle as pkl

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

def extract_features(model, x, phase):
    ### strip the last layer
    outputs = []
    with torch.no_grad():
        model.eval()
        for inputs, _ in tqdm(x[phase], desc = f"Extracting {phase} Features..."):
            inputs = inputs.to(device)
            output = model.get_embedding(inputs).data.cpu().numpy()
            outputs.append(output)
        print(outputs[0].shape)
        return outputs

BATCH_SIZE = 4
DATASET = '../../MIT_split'
#load data and transform to siamese format
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(os.path.join(DATASET,'train'))
test_dataset = torchvision.datasets.ImageFolder(os.path.join(DATASET,'test'))


siamese_train_dataset = SiameseNetworkDataset(train_dataset,transform) # Returns pairs of images and target same/different
siamese_test_dataset = SiameseNetworkDataset(test_dataset,transform)

siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

embedding_net = EmbeddingNet()

model = SiameseNet(embedding_net)

model.to(device)

margin = 1.
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 500

fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

torch.save(model.state_dict(), "./models/fine_tuned_siamese.pt")

train_dataset = torchvision.datasets.ImageFolder(os.path.join(DATASET,'train'), transform=transform)
test_dataset = torchvision.datasets.ImageFolder(os.path.join(DATASET,'test'), transform=transform)
# RESET LOADERS WITHOUT SHUFFLE
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)

dataloaders = {"train": train_dataloader, "test": test_dataloader}

train_features = extract_features(model,dataloaders,"train")
test_features = extract_features(model,dataloaders,"test")

pkl.dump(train_features, open("./features/train_features_siamese.pkl", "wb"))
pkl.dump(test_features, open("./features/test_features_siamese.pkl","wb"))

