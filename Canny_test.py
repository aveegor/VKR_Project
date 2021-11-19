import numpy as np
import cv2 as cv
from analisys_craks import Pic, Settings

# Canny settings
low_thr = 35
high_thr = low_thr * 3
apr_size = 3

# Morphology settings
kernel_3 = np.ones((3, 3), np.uint8)
kernel_5 = np.ones((5, 5), np.uint8)

img = cv.imread('sample_2.jpg')
test_img = Pic(img)

canny = cv.Canny(Pic.blur(test_img), low_thr, high_thr, apr_size)

# 1st method: dilate + close + erode
morph_img_1 = cv.dilate(canny, kernel_3, iterations=1)
morph_img_1 = cv.morphologyEx(morph_img_1, cv.MORPH_CLOSE, kernel_3, iterations=2)
morph_img_1 = cv.erode(morph_img_1, kernel_3, iterations=1)

# 2nd method: dilate + open + erode
# morph_img_2 = cv.dilate(canny, kernel_3, iterations=2)
# morph_img_2 = cv.morphologyEx(morph_img_2, cv.MORPH_OPEN, kernel_3, iterations=1)
# morph_img_2 = cv.erode(morph_img_2, kernel_3, iterations=1)

no_canny = cv.morphologyEx(Pic.adapt_thr(test_img), cv.MORPH_OPEN, kernel_3, iterations=1)

cv.imshow('thr', test_img.img)
cv.imshow('canny', canny)
cv.imshow('canny+morph', morph_img_1)
cv.waitKey(0)
cv.destroyAllWindows()
