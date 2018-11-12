#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 23:43:31 2018

@author: myidispg
"""

# import libraries
import torch
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms

# Number of subprocesses to use for data loading
num_workers = 0
# how many samples to load per batch
batch_size = 20

# convert data to torch.FloatTensor()
transform = transforms.ToTensor()

# choose the training and test dataset

train_data = datasets.MNIST(root='MNIST_data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='MNIST_data', train=True, download=True, transform=transform)

# Prepare the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

import matplotlib.pyplot as plt
%matplotlib inline

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch along with the corresponding labels
fig = plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    ax.set_title(str(labels[idx].item()))
    
# View an image in more detail
img  = np.squeeze(images[1])
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y], 2) if img[x][y] != 0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
        
# DEFINE THE NETOWRK ARCHITECTURE
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28*28)        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
# initialize the nn
model = Net()
# specify the loss function
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.03)

# number of epochs
epochs = 30

model.train() # set the model to train mode

for epoch in range(epochs):
    train_loss = 0.0
    
    for data, labels in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, 
        train_loss
        ))
    
# TEST THE TRAINED NETWORK
# initialize list to monitor test loss and accuracy    




