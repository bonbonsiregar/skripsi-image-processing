import cv2
import numpy as np

img = cv2.imread('0_1.jpg')
skel = np.zeros(img.shape,np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

cv2.imshow('img',img)

erode = cv2.erode(img,element)

cv2.imshow('erode',erode)

dilate = cv2.dilate(erode,element)

cv2.imshow('dilate',dilate)

subtract = cv2.subtract(img,dilate)

cv2.imshow('subtract',subtract)

bitwise_or = cv2.bitwise_or(skel,subtract)

cv2.imshow('bitwise_or',bitwise_or)

cv2.waitKey(0)
cv2.destroyAllWindows()