__author__ = 'mkv-aql'

#read all images in a folder, then resize them and save them in a new folder with its original names
import cv2
import matplotlib.pyplot as plt
import os



# read all images in a folder
images = []
filename_list = []

def get_image(folder):
    print("Running get_image")
    for filename in os.listdir(folder):
        filename_list.append(filename) #append the names of the data_preprocessing_scripts to a list
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    # print("Total elements: ", len(images)) #Just to check the no. of elements in the 'images' array
    return images


# resize all images in a folder
def resize_image(images):
    print("Running resize_image")
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (512, 512))
    return images

#list all the data_preprocessing_scripts in the folder
def list_files(folder):
    print("Running list_files")
    for filename in os.listdir(folder):
        print(filename)

# save all images with its original names in a new folder
def save_image(images):
    print("Running save_image")
    for i in range(len(images)):
        cv2.imwrite("C:/Users/Makav/Desktop/ISM_2022w/train_resized/" + str(i) + ".png", images[i])

# main function
folder = "C:/Users/Makav/Desktop/ISM_2022w/test"  # Simply change the file path to the correct location
get_image(folder)
filename_list.sort()
print(filename_list)
resize_image(images)
save_image(images)
