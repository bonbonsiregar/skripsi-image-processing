import cv2 
import numpy as np

img = cv2.imread('0_2.jpg')
n_white_pix = np.sum(img == 255)
print('Jumlah dari white pixels: ', n_white_pix)