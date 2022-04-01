import torch
from torchsummary import summary
import torch.nn as nn
import torchvision
import os
from tqdm import tqdm
import torchvision.transforms as transforms

dataset_path = '../MIT_split'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
model = torch.load('../week1/model_90.pth')
BATCH_SIZE=1
newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
#adding global average pooling bc the way we defined out model doesn't let us remove only the last layer bu the whole FC
newmodel = torch.nn.Sequential(newmodel, nn.AdaptiveAvgPool2d((1,1)))

transform = transforms.Compose([
    transforms.ToTensor()
])

#train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path,'train'))
test_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path,'test'),transform=transform)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8
)

size = len(test_dataloader.dataset)
num_batches = len(test_dataloader)
newmodel.eval()
results=[]
with torch.no_grad():
    for batch in tqdm(test_dataloader,desc='iter'):
        images = batch[0].to(device)  # Get batch of images from our train dataloader
        labels = batch[1].to(device)  # Get the corresponding target(0, 1 or 2) representing cats, dogs or pandas
        # calculate outputs by running images through the network
        outputs = newmodel(images)
        results.append(outputs)

