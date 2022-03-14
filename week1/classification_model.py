import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
"""
print('Using:', device)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
"""

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5), nn.ReLU()
        ).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(128, 1024), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024,8)
        ).to(device)

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x,(1,1))
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
