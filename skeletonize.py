import cv2
import numpy as np
 
img = cv2.imread('0_1.jpg',0)
size = np.size(img)
skel = np.zeros(img.shape,np.uint8)

element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
 
while(done is False):
    eroded = cv2.erode(img,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img,temp)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()
 
    zeros = size - cv2.countNonZero(img)
    if zeros==size:
        done = True
cv2.imshow("skel",skel)
cv2.imwrite('0_2.jpg',skel)
cv2.waitKey(0)
cv2.destroyAllWindows()
