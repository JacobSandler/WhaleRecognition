import torch
import torch.nn as nn
import torch.nn.functional as F

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):

        #maybe we should add this? Idk why it was there
        #super(CNN, self).init()
        super().__init__()
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=1) # in_channels = 3 bc RGB 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3) # in_channels comes from previous output
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=8, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5)
        self.conv6 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #in and out features will change with image size
        self.flat = nn.Flatten() # can use this instead of .view() function in forward()
        self.fc1 = nn.Linear(in_features=19558, out_features=5005) # output of 5005 because we have 5005 different whale ids

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool3(F.relu(self.conv6(x)))
        x = self.flat(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x