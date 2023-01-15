import numpy as np
import os
import cv2
#PATH_TRAIN = r"Sem-4/ISM/ISM_2022w/train"
#folder_dir = "Sem-4/ISM/ISM_2022w/train/"
PATH_TRAIN = r"C:/Users/Makav/Desktop/ISM_2022w/train"
folder_dir = "C:/Users/Makav/Desktop/ISM_2022w/train/"

for images in os.listdir(folder_dir):


         # check if the image ends with png
       if (images.endswith(".png")):
        image = cv2.imread(PATH_TRAIN + "/" + images, cv2.IMREAD_GRAYSCALE)
        #res_img = cv2.resize(image, (256, 256))
        res_img = cv2.resize(image, (512, 512))
        # Convert the array back to an image and save it as a JPEG
        cv2.imwrite("C:/Users/Makav/Desktop/ISM_2022w/train_resized/"+images, res_img)
