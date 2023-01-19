__author__ = 'mkv-aql'
import os
import cv2
import glob
import torch
import shutil
import itertools
import torch.nn as nn
import torch.optim as optim
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn import functional as F
from torchvision import datasets, models, transforms

"""
os.mkdir('/kaggle/working/train')
os.mkdir('/kaggle/working/test')

os.mkdir('/kaggle/working/train/covid')
os.mkdir('/kaggle/working/test/covid')

os.mkdir('/kaggle/working/train/normal')
os.mkdir('/kaggle/working/test/normal')

os.mkdir('/kaggle/working/train/pneumonia')
os.mkdir('/kaggle/working/test/pneumonia')
"""

normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        normalizer
    ]),

    'validation': transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        normalizer
    ])
}

data_images = {
    'train': datasets.ImageFolder('C:/Users/Makav/Desktop/ISM_2022w/resnet_train', data_transforms['train']),
    'validation': datasets.ImageFolder('C:/Users/Makav/Desktop/ISM_2022w/resnet_test', data_transforms['validation'])
}

dataloaders = {
    'train': torch.utils.data.DataLoader(data_images['train'], batch_size=32, shuffle=True, num_workers=0),
    'validation': torch.utils.data.DataLoader(data_images['validation'], batch_size=32,shuffle=True,num_workers=0)
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True) #This will automatically download the pretrained weights

#Freeze the weights of the pretrained model
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 4) #(Linear(n,m)) n = input size (no. of weights), m = output/neuron size, 4 = no. of classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())


def trained_model(model, criterion, optimizer, epochs):
    for epoch in range(epochs):

        print('Epoch:', str(epoch + 1) + '/' + str(epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # this trains the model
            else:
                model.eval()  # this evaluates the model

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # convert inputs to cpu or cuda
                labels = labels.to(device)  # convert labels to cpu or cuda

                outputs = model(inputs)  # outputs is inputs being fed to the model
                loss = criterion(outputs, labels)  # outputs are fed into the model

                if phase == 'train':
                    optimizer.zero_grad()  # sets gradients to zero
                    loss.backward()  # computes sum of gradients
                    optimizer.step()  # preforms an optimization step

                _, preds = torch.max(outputs, 1)  # max elements of outputs with output dimension of one
                running_loss += loss.item() * inputs.size(0)  # loss multiplied by the first dimension of inputs
                running_corrects += torch.sum(preds == labels.data)  # sum of all the correct predictions

            epoch_loss = running_loss / len(data_images[phase])  # this is the epoch loss
            epoch_accuracy = running_corrects.double() / len(data_images[phase])  # this is the epoch accuracy

            print(phase, ' loss:', epoch_loss, 'epoch_accuracy:', epoch_accuracy)

    return model

model = trained_model(model, criterion, optimizer, 3)

torch.save(model.state_dict(), 'C:/Users/Makav/Desktop/ISM_2022w/resnet_model/weights.h5') #save the model's weights
model.load_state_dict(torch.load('C:/Users/Makav/Desktop/ISM_2022w/resnet_model/weights.h5')) #load the model's weights