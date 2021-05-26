import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):

        #maybe we should add this? Idk why it was there
        #super(CNN, self).init()
        
        super().__init__()
    
        self.resnet =  torch.hub.load('pytorch/vision:v0.9.0','resnext101_32x8d',pretrained=True)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.resnet.eval()

        self.flat = nn.Flatten() # can use this instead of .view() function in forward()
        self.fc1 = nn.Linear(in_features=2048, out_features=5005) # output of 5005 because we have 5005 different whale ids

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x)

        x = self.flat(x)
        x = self.fc1(x)

        return x