from skimage import io
from skimage import filters
import matplotlib.pyplot as plt
import cv2
import numpy as np


image = plt.imread("0_1.jpg")
#asfloat = img_as_float(image)
binary = filters.threshold_otsu(image)


#plt.imshow()
print(type(binary))