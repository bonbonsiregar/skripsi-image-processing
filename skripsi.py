import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Image
image = cv2.imread("C:\\Users\\USER\\Documents\\skripsi_image_processing\\gambar meloidogyne\\Meloidogyne chitwoodi\melochit20_BP-01_NebraskaSupermarket-edit.jpg")
# Load Image untuk threshold
img = cv2.imread("C:\\Users\\USER\\Documents\\skripsi_image_processing\\gambar meloidogyne\\Meloidogyne chitwoodi\melochit20_BP-01_NebraskaSupermarket-edit.jpg")
# Matrix untuk Morphological closing
kernel_closing = np.ones((15,15),np.uint8)
# Matrix untik Morphological opening
kernel_opening = np.ones((3,3),np.uint8)
# konversi gambar berwarna 3 color channel ke grayscale dengan 1 color channel
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gaussian blur dengan kernel matrix 3x3
gaussian_blur = cv2.GaussianBlur(gray_img,(3, 3),0)

ret, threshold_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)

def auto_canny(image, sigma=0.33):
    
    v = np.median(image)
    
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 

    return edged
# Canny manual dengan jarak value yang lebar
wide = cv2.Canny(gaussian_blur,10,200)
# Canny manual dengan jarak value yang kecil
tight = cv2.Canny(gaussian_blur,225,250)
# Auto Canny Edge Detection
auto = auto_canny(gaussian_blur)

# Melakukan morphological closing untuk mengisi bagian dalam objek 
closing = cv2.morphologyEx(wide, cv2.MORPH_CLOSE,kernel_closing)

# Melakukan morphological opening untuk menghapus noise di luar objek
opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel_opening)


#cv2.drawContours(img, contours, -1, (255,255,0),1)
cv2.imwrite('0_1.jpg',opening)

caption = ['image', 'grayscale', 'gaussian blur', 'canny','threshold', 'closing', 'opening']
images = [image,gray_img,gaussian_blur,auto,threshold_img,closing,opening]

for i in range(7):
     plt.subplot(3,3, i+1), plt.imshow(images[i], 'gray')
     plt.title(caption[i])
     plt.xticks([]),plt.yticks([])
    
plt.show()

#cv2.imshow('image', image);
#cv2.imshow('grayscale', gray_img);
#cv2.imshow('gaussian blur', gaussian_blur);
cv2.imshow('canny', wide);
#cv2.imshow('threshold', threshold_img);
#cv2.imshow('closing', closing);
cv2.imshow('opening', opening);
cv2.waitKey(0)
cv2.destroyAllWindows()
