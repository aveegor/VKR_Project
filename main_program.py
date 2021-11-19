import numpy as np
import cv2 as cv
from analisys_craks import Pic, Settings


def elongation(moments):
    x = moments['mu20'] + moments['mu02']
    y = 4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2
    return (x + y ** 0.5) / (x - y ** 0.5)


def find_dark_cnts(contours):
    dark_contours = []
    mask = np.zeros(Pic.gray(crack).shape, np.uint8)
    for contour in contours:
        pseudo_mask = cv.drawContours(mask.copy(), contour, -1, 255, -1)
        mean = cv.mean(Pic.gray(crack), mask=pseudo_mask)
        if mean[0] <= crack.settings.thr_lower_limit:
            dark_contours.append(contour)
    return dark_contours


def find_elongated_cnts(contours):
    elongated_contours = []
    mask = np.zeros(Pic.gray(crack).shape, np.uint8)
    for contour in contours:
        pseudo_mask = cv.fillPoly(mask.copy(), [contour], color=(255, 255, 255))
        moments_of_mask = cv.moments(pseudo_mask)
        if elongation(moments_of_mask) >= crack.settings.lower_limit_elongation:
            elongated_contours.append(contour)
    return elongated_contours


img_1 = cv.imread('sample_7.jpg')
crack = Pic(img_1)

contours, _ = cv.findContours(Pic.mask_dark(crack), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

dark_contours = find_dark_cnts(contours)
elongated_contours = find_elongated_cnts(dark_contours)

elongated_contours_img = cv.drawContours(Pic.res(crack), elongated_contours, -1, (0, 255, 0), -1)

cv.imshow('img', Pic.res(crack))
cv.imshow('elon', elongated_contours_img)
cv.waitKey(0)
cv.destroyAllWindows()
