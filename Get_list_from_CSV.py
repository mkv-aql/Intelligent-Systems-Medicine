__author__ = 'mkv-aql'

#get data from csv file
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#Fetch data from csv file

def get_data_from_csv():
    print("Running get_data_from_csv")
    with open('C:/Users/Makav/Desktop/ISM_2022w/train_nooutput.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

data = get_data_from_csv()
print(data)
