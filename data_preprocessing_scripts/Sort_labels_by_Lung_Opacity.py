__author__ = 'mkv-aql'

import csv
import cv2
import os
"""
#name = input("000a3ae1-9927-4494-8933-01439b39a3c4.png")
with open("C:/Users/Makav/Desktop/ISM_2022w/train_original.csv", 'r') as file:
    reader = csv.reader(file, delimiter=' ')
    for row in reader:
        print(row[0])
        print(row[1])
"""
folder_dir = "C:/Users/AGAM MUHAJIR/Desktop/ISM_2022w/train_resized/"
PATH_TRAIN = r"C:/Users/AGAM MUHAJIR/Desktop/ISM_2022w/train_resized"
##Very slow and inefficient, but it works
with open("C:/Users/AGAM MUHAJIR/Desktop/ISM_2022w/train_original_Lung_Opacity.csv", 'r') as file:
    reader = csv.reader(file, delimiter=' ')
    for row in reader:
        for images in os.listdir(folder_dir):
            if (images.endswith(".png")):
                image = cv2.imread(PATH_TRAIN + "/" + images, cv2.IMREAD_GRAYSCALE)
                if row[0] == images:
                    print(row[1])
                    if row[1] == "Lung_Opacity":
                        cv2.imwrite("C:/Users/AGAM MUHAJIR/Desktop/ISM_2022w/train_lung_opacity/" + images, image)
                        break