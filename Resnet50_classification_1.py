__author__ = 'mkv-aql'

import os
import cv2
from glob import glob
import torch
import shutil
import itertools
import torch.nn as nn
import torch.optim as optim
import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn import functional as F
from torchvision import datasets, models, transforms

#Path to the data of each label
path = "C:/Users/Makav/Desktop/ISM_2022w"
covid_path = "C:/Users/Makav/Desktop/ISM_2022w/train_covid"
normal_path = "C:/Users/Makav/Desktop/ISM_2022w/train_normal"
pneumonia_path = "C:/Users/Makav/Desktop/ISM_2022w/train_pneumonia"
lung_opacity_path = "C:/Users/Makav/Desktop/ISM_2022w/train_lung_opacity"

images = sorted(glob(os.path.join(path, "train_resized", "*.png"))) #list of all images

fig, axes = plt.subplots(4, 4, figsize=(7, 7)) #4x4 grid of images
labels = ['Covid', 'Normal', 'Pneumonia', 'Lung Opacity']
i = 0
j = 0

for row in axes:
    for plot in row:
        plot.imshow(cv2.imread(images[j], 0))
        plot.axhline(y=0.5, color='r')
        plot.set_title(labels[i], fontsize=15)
        plot.axis('off')
        i += 1
        j += 1
    i = 0

fig.tight_layout()
plt.show()

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

#Create 80:20 train-test split
covid_train_len = int(np.floor(len(os.listdir(covid_path))*0.8))
covid_len = len(os.listdir(covid_path))

normal_train_len = int(np.floor(len(os.listdir(normal_path))*0.8))
normal_len = len(os.listdir(normal_path))

pneumonia_train_len = int(np.floor(len(os.listdir(pneumonia_path))*0.8))
pneumonia_len = len(os.listdir(pneumonia_path))

lung_opacity_train_len = int(np.floor(len(os.listdir(lung_opacity_path))*0.8))
lung_opacity_len = len(os.listdir(lung_opacity_path))

"""glo.iglob() does not work, therefore changed to glob()"""
for trainimg in itertools.islice(glob(os.path.join(covid_path, '*.png')), covid_train_len):
    shutil.copy(trainimg, 'C:/Users/Makav/Desktop/ISM_2022w/resnet_train/train_covid2')

for trainimg in itertools.islice(glob(os.path.join(normal_path, '*.png')), normal_train_len):
    shutil.copy(trainimg, 'C:/Users/Makav/Desktop/ISM_2022w/resnet_train/train_normal2')

for trainimg in itertools.islice(glob(os.path.join(pneumonia_path, '*.png')), pneumonia_train_len):
    shutil.copy(trainimg, 'C:/Users/Makav/Desktop/ISM_2022w/resnet_train/train_pneumonia2')

for trainimg in itertools.islice(glob(os.path.join(lung_opacity_path, '*.png')), lung_opacity_train_len):
    shutil.copy(trainimg, 'C:/Users/Makav/Desktop/ISM_2022w/resnet_train/train_lung_opacity2')


for testimg in itertools.islice(glob(os.path.join(covid_path, '*.png')), covid_train_len, covid_len):
    shutil.copy(testimg, 'C:/Users/Makav/Desktop/ISM_2022w/resnet_test/test_covid')

for testimg in itertools.islice(glob(os.path.join(normal_path, '*.png')), normal_train_len, normal_len):
    shutil.copy(testimg, 'C:/Users/Makav/Desktop/ISM_2022w/resnet_test/test_normal')

for testimg in itertools.islice(glob(os.path.join(pneumonia_path, '*.png')), pneumonia_train_len, pneumonia_len):
    shutil.copy(testimg, 'C:/Users/Makav/Desktop/ISM_2022w/resnet_test/test_pneumonia')

for trainimg in itertools.islice(glob(os.path.join(lung_opacity_path, '*.png')), lung_opacity_train_len, lung_opacity_len):
    shutil.copy(trainimg, 'C:/Users/Makav/Desktop/ISM_2022w/resnet_test/test_lung_opacity')
