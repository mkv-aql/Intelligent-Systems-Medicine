__author__ = 'mkv-aql'
#importing the necessary libraries
import numpy as np
import cv2
import os
"""
 #reading the X-ray image
img = cv2.imread('C:/Users/Makav/Desktop/ISM_2022w/train_left_512x512/00a3cd72-9d19-44dd-8064-9a53d32c731c.png')
 #converting the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 #applying a Gaussian blur
blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)
 #applying an Otsu threshold
thresh_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_OTSU)[1]
 #finding the contours of the image
contours = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 #creating a mask for the lung
mask = np.zeros(img.shape[:2], dtype="uint8")
 #iterating over the contours
for c in contours[0]:
	#if the contour is large enough, draw it on the mask
	if cv2.contourArea(c) > 2000:
		cv2.drawContours(mask, [c], -1, 255, -1)
 #applying the mask to the image
masked_image = cv2.bitwise_and(img, img, mask=mask)
 #showing the image
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
"""
#Left side
#PATH_TRAIN = r"C:/Users/Makav/Desktop/ISM_2022w/train_left_512x512"
#folder_dir = "C:/Users/Makav/Desktop/ISM_2022w/train_left_512x512/"
#right side
PATH_TRAIN = r"C:/Users/Makav/Desktop/ISM_2022w/train_right_512x512"
folder_dir = "C:/Users/Makav/Desktop/ISM_2022w/train_right_512x512/"

for images in os.listdir(folder_dir):
         # check if the image ends with png
       if (images.endswith(".png")):
        #image = cv2.imread(PATH_TRAIN + "/" + images, cv2.IMREAD_GRAYSCALE) #Cannot be grayscaled here
        image = cv2.imread(PATH_TRAIN + "/" + images)
        # converting the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # applying a Gaussian blur
        blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # applying an Otsu threshold
        thresh_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_OTSU)[1]
        # finding the contours of the image
        contours = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # creating a mask for the lung
        mask = np.zeros(image.shape[:2], dtype="uint8")
        # iterating over the contours
        for c in contours[0]:
            # if the contour is large enough, draw it on the mask
            if cv2.contourArea(c) > 2000:
                cv2.drawContours(mask, [c], -1, 255, -1)
        # applying the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)


        # Convert the array back to an image and save it as a original format image
        #left side
        #cv2.imwrite("C:/Users/Makav/Desktop/ISM_2022w/train_left_512x512_masked/"+images, masked_image)
        #right side
        cv2.imwrite("C:/Users/Makav/Desktop/ISM_2022w/train_right_512x512_masked/"+images, masked_image)