## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input image size 1*224*224
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv1_drop = nn.Dropout2d(p=0.1)
        # output size for each image at this point 32*110*110
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2_drop = nn.Dropout2d(p=0.2)
        # output size for each image at this point 64*54*54
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv3_drop = nn.Dropout2d(p=0.3)
        # output size for each image at this point 128*26*26
        
        self.fc1 = nn.Linear(128*26*26, 1000)
        self.fc1_drop = nn.Dropout(p=0.4)
        
        self.fc2 = nn.Linear(1000, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.conv1_drop(self.pool1(F.relu(self.conv1(x))))
        x = self.conv2_drop(self.pool2(F.relu(self.conv2(x))))
        x = self.conv3_drop(self.pool3(F.relu(self.conv3(x))))
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1_drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
