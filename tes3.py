import cv2
import numpy as np
 
img = cv2.imread('0_1.jpg',0)
size = np.size(img)
skel = np.zeros(img.shape,np.uint8)

ret,img = cv2.threshold(img,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

eroded = cv2.erode(img,element)
cv2.imshow("eroded",eroded)
temp = cv2.dilate(eroded,element)
cv2.imshow("temp",temp)
temp = cv2.subtract(img,temp)
cv2.imshow("temp",temp)
skel = cv2.bitwise_or(skel,temp)
cv2.imshow("skel",skel)

a = cv2.countNonZero(img)
print(a)
print(size)
cv2.waitKey(0)
cv2.destroyAllWindows()