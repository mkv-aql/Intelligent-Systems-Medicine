__author__ = 'mkv-aql'
from cv2 import cv2
import os

# Read the image
img = cv2.imread('C:/Users/Makav/Desktop/ISM_2022w/train_resized/000a3ae1-9927-4494-8933-01439b39a3c4.png', 0)
himg, wimg = img.shape

# Cut image in half
width_cutoff = wimg // 2
s1 = img[:, :width_cutoff]
s2 = img[:, width_cutoff:]

# Get size of cut image
hs1, ws1 = s1.shape
hs2, ws2 = s2.shape

print("Size of left image: ", hs1, ws1)
print("Size of right image: ", hs2, ws2)

# Show the cut off image
cv2.imshow('Match', s1)
cv2.waitKey(0)
cv2.imshow('Match', s2)
cv2.waitKey(0)
cv2.destroyAllWindows()


PATH_TRAIN = r"C:/Users/Makav/Desktop/ISM_2022w/train_resized"
folder_dir = "C:/Users/Makav/Desktop/ISM_2022w/train_resized/"

#Getting the correct size, since .shape cannot work with the for loop
#images = cv2.imread('C:/Users/Makav/Desktop/ISM_2022w/train_resized/000a3ae1-9927-4494-8933-01439b39a3c4.png', 0)
#himg, wimg = images.shape

for images in os.listdir(folder_dir):
         # check if the image ends with png
       if (images.endswith(".png")):
        image = cv2.imread(PATH_TRAIN + "/" + images, cv2.IMREAD_GRAYSCALE)
        himg, wimg = image.shape

        # Cut image in half
        width_cutoff = wimg // 2
        s1 = image[:, :width_cutoff]
        s2 = image[:, width_cutoff:]
        # Convert the array back to an image and save it as a original format image
        cv2.imwrite("C:/Users/Makav/Desktop/ISM_2022w/train_left_512x512/"+images, s1)
        cv2.imwrite("C:/Users/Makav/Desktop/ISM_2022w/train_right_512x512/"+images, s2)

