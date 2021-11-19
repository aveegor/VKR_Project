import numpy as np
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import time


def draw_borders(local_rectangles_img, size_patch, end_of_column, end_of_row, cnt_patches):
    start_point = (end_of_column - size_patch[0], end_of_row - size_patch[1])
    end_point = (end_of_column, end_of_row)
    local_rectangles_img = cv.rectangle(local_rectangles_img, start_point, end_point,
                                        (0, 0, 0), 1)
    local_rectangles_img = cv.putText(local_rectangles_img, str(cnt_patches),
                                      (end_of_column - size_of_patch[0], end_of_row - size_of_patch[1] // 2),
                                      cv.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv.LINE_AA)
    return local_rectangles_img


def find_patches(img, size_patch, grid_flag=False):
    img_size = [img.shape[1], img.shape[0]]
    local_patches = []
    local_rectangles_img = img.copy()
    cnt_patches = 0
    for end_of_row in range(size_patch[1], img_size[1] + size_patch[1], size_patch[1]):
        for end_of_column in range(size_patch[0], img_size[0] + size_patch[0], size_patch[0]):
            local_patches.append(img[end_of_row - size_patch[1]:end_of_row,
                                 end_of_column - size_patch[0]:end_of_column])  # input_img[height, width]
            # draw borders of patches
            if grid_flag:
                local_rectangles_img = draw_borders(local_rectangles_img, size_patch, end_of_column,
                                                    end_of_row, cnt_patches)
            else:
                local_rectangles_img = None
            cnt_patches += 1
    if grid_flag:
        return local_patches, local_rectangles_img
    else:
        return local_patches


def find_spectrum(img):
    img_spectrum = [[], [], []]
    for row_pix in range(img.shape[0]):
        for column_pix in range(img.shape[1]):
            img_spectrum[0].append(img[row_pix, column_pix][0])  # Blue spectrum
            img_spectrum[1].append(img[row_pix, column_pix][1])  # Green spectrum
            img_spectrum[2].append(img[row_pix, column_pix][2])  # Red spectrum
    return img_spectrum


def nrm_clrs_val(img_spectrum):
    normalized_value = [[], [], []]
    number_of_pixels = len(img_spectrum[0])
    for i in range(0, 256):
        normalized_value[0].append(img_spectrum[0].count(i) / number_of_pixels)  # Blue spectrum
        normalized_value[1].append(img_spectrum[1].count(i) / number_of_pixels)  # Green spectrum
        normalized_value[2].append(img_spectrum[2].count(i) / number_of_pixels)  # Red spectrum
    return normalized_value


# settings

size_of_patch = [20, 20]  # [width, height]
size_of_img = [1920, 1080]
bins = 125
patch_number = 3591

diff_patches = [[], [], []]
with open('corrosion_samples.json', 'r') as file:
    spectrum_samples = json.load(file)

if __name__ == '__main__':

    input_img = cv.imread('corrosion_3.jpg')
    input_img = cv.resize(input_img, (size_of_img[0], size_of_img[1]), interpolation=cv.INTER_AREA)
    input_img = cv.GaussianBlur(input_img, [3, 3], 0)
    s = time.time()
    patches = find_patches(input_img, size_of_patch)
    spectrum = find_spectrum(patches[patch_number])
    spectrum = nrm_clrs_val(spectrum)
    e = time.time()
    print(e - s)

    # cv.imshow('img', rectangles_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # fig1, (blue_spectrum, green_spectrum, red_spectrum) = plt.subplots(1, 3, figsize=(12, 6))
    # blue_spectrum.hist(patch_spectrum[0], bins, range=(0, 256), color='b', density=True)
    # green_spectrum.hist(patch_spectrum[1], bins, range=(0, 256), color='g', density=True)
    # red_spectrum.hist(patch_spectrum[2], bins, range=(0, 256), color='r', density=True)
    # blue_spectrum.grid(True)
    # green_spectrum.grid(True)
    # red_spectrum.grid(True)
    # plt.show()
