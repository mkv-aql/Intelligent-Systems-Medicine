__author__ = 'mkv-aql'
import cv2
import numpy as np
 #load image
image = cv2.imread('C:/Users/Makav/Desktop/ISM_2022w/train_left_512x512/00a29457-e620-4935-8bec-2b88855d385d.png')
 #convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 #apply thresholding
threshold_value, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 #apply morphological operations
kernel = np.ones((3,3), np.uint8)
closing = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel, iterations=1)
 #find contours
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 #iterate through contours
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    #draw rectangles around contours
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
 #show result image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
