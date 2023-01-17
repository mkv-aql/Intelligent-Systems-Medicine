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

#Path to the data of each label
covid_path = '../input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19'
normal_path = '../input/covid19-radiography-database/COVID-19 Radiography Database/NORMAL'
pneumonia_path = '../input/covid19-radiography-database/COVID-19 Radiography Database/Viral Pneumonia'
lung_opacity_path = '../input/covid19-radiography-database/COVID-19 Radiography Database/Lung_Opacity'