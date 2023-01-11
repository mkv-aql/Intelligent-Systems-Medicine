__author__ = 'mkv-aql'

import cv2
import matplotlib.pyplot as plt
import os


#read image
img = cv2.imread('Dataset_experiment/a.png')
#print image width, height and channels
print(img.shape)
#show image
plt.imshow(img)
plt.show()

#resize image
img = cv2.resize(img, (512, 512))
#print image width, height and channels
print(img.shape)
#show image
plt.imshow(img)
plt.show()

