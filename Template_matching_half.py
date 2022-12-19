__author__ = 'mkv-aql'

import numpy as np
from cv2 import cv2


img = cv2.imread('test/0a23fc8b-01c1-4f0b-a33c-d749811da434.png', 0) #Load image as grayscale
template = cv2.imread('Assets/Unbenannt.JPG', 0)
h, w = template.shape
himg, wimg = img.shape

# Cut image in half
width_cutoff = wimg // 2
s1 = img[:, :width_cutoff]
s2 = img[:, width_cutoff:]

#Get size of cut image
hs2, ws2 = s2.shape

#Show the cut off image
cv2.imshow('Match', s1)
cv2.waitKey(0)
cv2.imshow('Match', s2)
cv2.waitKey(0)
cv2.destroyAllWindows()

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img.copy()

    #Left side of image search
    result = cv2.matchTemplate(s1, template, method)
    #(W - w + 1, H - h + 1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) #Variables are in tuples
    print("Left coordinate: ", min_loc, max_loc)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    # Right side of image search
    result2 = cv2.matchTemplate(s2, template, method)
    # (W - w + 1, H - h + 1)
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result2) #Variables are in tuples
    #Convert tuples to list
    list_min_loc2 = list(min_loc2)
    list_max_loc2 = list(max_loc2)
    #Add width value to the list
    list_min_loc2[0] = list_min_loc2[0] + ws2
    list_max_loc2[0] = list_max_loc2[0] + ws2
    #print(min_loc2, max_loc2)
    print("Right coordinate: ", list_min_loc2, list_max_loc2)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location2 = list_min_loc2
    else:
        location2 = list_max_loc2

    #Show left side
    bottom_right = (location[0] + w, location[1] + h)
    cv2.rectangle(img2, location, bottom_right, 255, 2)
    #Show righ side
    bottom_right2 = (location2[0] + w, location2[1] + h)
    cv2.rectangle(img2, location2, bottom_right2, 255, 2)

    cv2.imshow('Match', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
