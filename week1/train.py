import os
from classification_model import MyModel
import torch
import torch.nn as nn
import wandb
import torchvision
import torchvision.transforms as transforms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using:', device)


epochs = 200
BATCH_SIZE = 4
lr = 1e-3

wandb.init(project="M5-VisualRecognition", entity="m5-group6")

wandb.config = {
  "learning_rate": lr,
  "epochs": epochs,
  "batch_size": BATCH_SIZE,
  "architecture" : "CNN",
}
def get_dataloaders(BATCH_SIZE):


    dataset_path = '../MIT_split'

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


    # Transformation function to be applied on images
    # 1. Horizontally Flip the image with a probability of 30%
    # 5. Convert Image to a Pytorch Tensor
    # 6. Normalize the pytorch's tensor using mean & std of imagenet
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Create a dataset by from the dataset folder by applying the above transformation.
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path,'train'), transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path,'test'), transform=transform)


    # Create a Train DataLoader using Train Dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )
    # Create a Test DataLoader using Test Dataset
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )

    return train_dataloader, test_dataloader


def Train(model,train_dataloader):
    size = len(train_dataloader.dataset)
    model.train()
    loss_metric, correct = 0, 0
    for batch in train_dataloader:
        minput = batch[0].to(device)  # Get batch of images from our train dataloader
        target = batch[1].to(device)  # Get the corresponding target(0, 1 or 2) representing cats, dogs or pandas

        optimizer.zero_grad()  # Clear the gradients if exists. (Gradients are used for back-propogation.)

        moutput = model(minput)  # output by our model

        loss = criterion(moutput, target)  # compute cross entropy loss

        loss.backward()  # Back propogate the losses
        optimizer.step()  # Update Model parameters

        loss_metric += loss.item()
        correct += (moutput.argmax(1) == target).type(torch.float).sum().item()

    loss = loss_metric / len(train_dataloader)
    acc = correct / size
    print(f"Training loss: {loss:>7f}, Training accuracy: {acc:>7f}")
    return loss, acc


def Test(model, test_dataloader):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    loss_metric_test, correct = 0, 0
    with torch.no_grad():
        for batch in test_dataloader:
            images = batch[0].to(device)  # Get batch of images from our train dataloader
            labels = batch[1].to(device)  # Get the corresponding target(0, 1 or 2) representing cats, dogs or pandas
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            loss_metric_test += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
    loss_metric_test /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss_metric_test:>8f} \n")

    return loss_metric_test, correct

train_dataloader, test_dataloader = get_dataloaders(BATCH_SIZE)

model = MyModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
wandb.watch(model)


for epoch in range(0, epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train_loss, train_acc = Train(model,train_dataloader)
    test_loss, test_acc = Test(model,test_dataloader)

    wandb.log({"Train loss": train_loss,
               "Train accuracy": train_acc,
               "Valid loss": test_loss,
               "Valid accuracy": test_acc, "epoch": epoch})
    print('\n')

    if epoch % 10 == 0:
        torch.save(model, 'model_' + str(epoch) + '.pth')

