import numpy as np
import cv2 as cv


class Settings:
    def __init__(self):
        self.resolution = [1280, 720]
        self.gauss_blr = [5, 5]
        self.thr_lower_limit = 105
        self.thr_upper_limit = 255
        self.thr_gauss_1th_coef = 53
        self.thr_gauss_2nd_coef = 11
        self.min_area = 100
        self.lower_limit_elongation = 30


class Pic:
    def __init__(self, img):
        self.img = img
        self.settings = Settings()

    def res(self):
        resized = cv.resize(self.img, (self.settings.resolution[0], self.settings.resolution[1]),
                            interpolation=cv.INTER_AREA)
        return resized

    def gray(self):
        gray = cv.cvtColor(self.res(), cv.COLOR_BGR2GRAY)
        return gray

    def blur(self):
        blur = cv.GaussianBlur(self.gray(), (self.settings.gauss_blr[0], self.settings.gauss_blr[1]), 0)
        return blur

    def thr(self):
        _, thr = cv.threshold(self.blur(), self.settings.thr_lower_limit, self.settings.thr_upper_limit,
                              cv.THRESH_TOZERO)
        return thr

    def adapt_thr(self):
        adapt_thr = cv.adaptiveThreshold(self.blur(), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
                                         self.settings.thr_gauss_1th_coef, self.settings.thr_gauss_2nd_coef)
        adapt_thr = cv.copyMakeBorder(adapt_thr, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, (0, 0, 0))
        return adapt_thr

    def contours(self):
        contours_all, _ = cv.findContours(self.adapt_thr(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = []
        for contour in contours_all:
            if cv.contourArea(contour) >= self.settings.min_area:
                contours.append(contour)
        return contours

    def number_of_contours(self):
        number_of_contours = len(self.contours())
        return number_of_contours

    def contour_area(self):
        contours = self.contours()
        contour_area = []
        for contour in contours:
            if cv.contourArea(contour) >= self.settings.min_area:
                contour_area.append(cv.contourArea(contour))
        contour_area.sort(reverse=True)
        contour_area = contour_area[:-1]
        return contour_area

    def total_area(self):
        total_area = sum(self.contour_area())
        return total_area

    def average_area(self):
        average_area = self.total_area() / self.number_of_contours()
        return average_area

    def percent_of_area(self):
        contour_area = self.contour_area()
        total_area = self.total_area()
        percent_of_area = []
        for area in contour_area:
            percent_of_area.append(area / total_area * 100)
        return percent_of_area

    def percent_log(self):
        contour_area = self.contour_area()
        percent_of_area = self.percent_of_area()
        for i in range(5):
            print(contour_area[i], float('{:.3f}'.format(percent_of_area[i])), '%')
        print()

    def draw_contours(self, cnt_for_draw):
        img_with_contours = cv.copyMakeBorder(self.img, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, (0, 0, 0))
        contours = self.contours()
        cv.drawContours(img_with_contours, contours, cnt_for_draw, (0, 0, 255), 1)
        return img_with_contours

    def mask_dark(self):
        mask = np.zeros(self.gray().shape, np.uint8)
        mask = cv.fillPoly(mask, self.contours(), color=(255, 255, 255))
        return mask

    def betwise_gray(self):
        img_thresh_gray = cv.bitwise_and(self.gray(), self.mask_dark())
        return img_thresh_gray
