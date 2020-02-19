import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("gambar meloidogyne/Meloidogyne hapla/melhap49_CT-04 1-edit.jpg")
img = cv2.imread("gambar meloidogyne/Meloidogyne hapla/melhap49_CT-04 1-edit.jpg")
kernel_closing = np.ones((20,20),np.uint8)
kernel_opening = np.ones((5,5),np.uint8)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gaussian_blur = cv2.GaussianBlur(gray_img,(3, 3),0)

ret, threshold_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)

def auto_canny(image, sigma=0.33):
    
    v = np.median(image)
    
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 

    return edged

wide = cv2.Canny(gaussian_blur,10,200)
tight = cv2.Canny(gaussian_blur,225,250)
auto = auto_canny(gaussian_blur)

#opening = cv2.morphologyEx(auto,cv2.MORPH_OPEN,kernel_opening)
closing = cv2.morphologyEx(auto, cv2.MORPH_CLOSE,kernel_closing)
opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel_opening)
#closing2 = cv2.morphologyEx(closing, cv2.MORPH_CLOSE,kernel)
#ret, threshold_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)



contours, hier = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:

    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)

    box = np.int0(box)

    cv2.drawContours(img,[box],0,(0,0,255))

    (x,y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius)

    img = cv2.circle(img,center,radius,(255,0,0),2)


#print jumlah contours yang terdeteksi
print(len(contours))

cv2.drawContours(img, contours, -1, (255,255,0),1)
cv2.imwrite('0_1.jpg',opening)

caption = ['image', 'grayscale', 'gaussian blur', 'canny', 'contours', 'threshold', 'opening', 'closing']
images = [image,gray_img,gaussian_blur,auto,img,threshold_img,opening,closing]

# for i in range(8):
#     plt.subplot(3,3, i+1), plt.imshow(images[i], 'gray')
#     plt.title(caption[i])
#     plt.xticks([]),plt.yticks([])
    
# plt.show()

cv2.imshow('opening', opening);
cv2.waitKey(0)
cv2.destroyAllWindows()
