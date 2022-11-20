__author__ = 'mkv-aql'

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy

def get_image():
    #grabbed_image = Image.open("Dataset/a.png") #Using PIL library
    print("running get_image")
    grabbed_image = mpimg.imread('Dataset/a.png')
    grabbed_image2 = mpimg.imread('Dataset/b.png')
    plt.imshow(grabbed_image)
    plt.show()
    #grabbed_image_test = ImageTk.PhotoImage(grabbed_image)

def show_image():
    print("nothing")

