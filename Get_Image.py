__author__ = 'mkv-aql'

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy

import cv2
import os

''' #matplotlib library method

def get_image():
    #grabbed_image = Image.open("Dataset/a.png") #Using PIL library
    print("running get_image")
    grabbed_image = mpimg.imread('Dataset/a.png')
    grabbed_image2 = mpimg.imread('Dataset/b.png')
    plt.imshow(grabbed_image)
    plt.show()
    #grabbed_image_test = ImageTk.PhotoImage(grabbed_image)
'''

def show_image():
    print("nothing")

#opencv library method
images = []
def get_image(folder):
    print("Running get_image")
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    print("Total elements: ", len(images)) #Just to check the no. of elements in the 'images' array
    return images

