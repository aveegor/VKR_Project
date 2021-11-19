import numpy as np
import cv2.cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import time


def draw_borders(local_rectangles_img, coordinates_of_patches):
    for patch in coordinates_of_patches:
        start_point = coordinates_of_patches[patch][0]
        end_point = coordinates_of_patches[patch][1]
        local_rectangles_img = cv.rectangle(local_rectangles_img, start_point, end_point,
                                            (0, 0, 0), 1)
        local_rectangles_img = cv.putText(local_rectangles_img, patch,
                                          (start_point[0], end_point[1] - ((end_point[1] - start_point[1]) // 2)),
                                          cv.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv.LINE_AA)
    return local_rectangles_img


def find_patches(img, size_patch, grid_flag=False):
    coordinates_of_patches = {}  # {'number_of_patch': [[upper left point], [lower right point]], ...}
    img_size = [img.shape[1], img.shape[0]]
    local_patches = []
    local_rectangles_img = None
    cnt_patches = 0
    for end_of_row in range(size_patch[1], img_size[1] + size_patch[1], size_patch[1]):
        for end_of_column in range(size_patch[0], img_size[0] + size_patch[0], size_patch[0]):
            start_of_row = end_of_row - size_patch[1]
            start_of_column = end_of_column - size_patch[0]
            local_patches.append(
                img[start_of_row:end_of_row, start_of_column:end_of_column])  # input_img[height, width]

            if grid_flag:
                coordinates_of_patches[str(cnt_patches)] = [[start_of_column, start_of_row],
                                                            [end_of_column, end_of_row]]

            cnt_patches += 1

    if grid_flag:
        local_rectangles_img = draw_borders(img.copy(), coordinates_of_patches)
    if grid_flag:
        return np.asarray(local_patches), local_rectangles_img
    else:
        return np.asarray(local_patches)


def find_intensity_bgr(img):
    channel_intensity = [[], [], []]
    for row_pix in range(0, img.shape[0]):
        for column_pix in range(0, img.shape[1]):
            channel_intensity[0].append(img[row_pix, column_pix][0])  # Blue channel
            channel_intensity[1].append(img[row_pix, column_pix][1])  # Green channel
            channel_intensity[2].append(img[row_pix, column_pix][2])  # Red channel
    return np.asarray(channel_intensity)


def find_spectrum(img, normalization_flag=False):
    channel_intensity = find_intensity_bgr(img)
    spectrum = np.zeros((3, 256))
    if normalization_flag:
        number_of_pixels = len(channel_intensity[0])
    else:
        number_of_pixels = 1
    for i in range(256):
        spectrum[0][i] = channel_intensity[0][channel_intensity[0] == i].size / number_of_pixels
        spectrum[1][i] = channel_intensity[1][channel_intensity[1] == i].size / number_of_pixels
        spectrum[2][i] = channel_intensity[2][channel_intensity[2] == i].size / number_of_pixels
    return spectrum


# settings
size_of_patch = [20, 20]  # [width, height]
size_of_img = [1920, 1080]
bins = 125
patch_number = 1264

diff_patches = [[], [], []]
with open('corrosion_samples.json', 'r') as file:
    spectrum_samples = json.load(file)
    spectrum_samples = spectrum_samples['samples']
    spectrum_samples_val = []
    for sample in spectrum_samples:
        spectrum_samples_val.append([spectrum_samples[sample]['blue_spm'],
                                    spectrum_samples[sample]['green_spm'],
                                    spectrum_samples[sample]['red_spm']])
spectrum_samples_val = np.asarray(spectrum_samples_val)


if __name__ == '__main__':
    input_img = cv.imread('corrosion_3.jpg')
    input_img = cv.resize(input_img, (size_of_img[0], size_of_img[1]), interpolation=cv.INTER_AREA)
    input_img = cv.GaussianBlur(input_img, [3, 3], 0)

    s = time.time()
    patches = find_patches(input_img, size_of_patch)
    for patch in patches:
        find_spectrum(patch, True) - spectrum_samples_val[0]
    e = time.time()
    print(e - s)

    # rectangles_img = cv.resize(rectangles_img, (1280, 720), interpolation=cv.INTER_AREA)
    # cv.imshow('img', rectangles_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    #
    # fig1, (blue_spectrum, green_spectrum, red_spectrum) = plt.subplots(1, 3, figsize=(12, 6))
    # blue_spectrum.hist(patch_spectrum[0], bins, range=(0, 256), color='b', density=False)
    # green_spectrum.hist(patch_spectrum[1], bins, range=(0, 256), color='g', density=False)
    # red_spectrum.hist(patch_spectrum[2], bins, range=(0, 256), color='r', density=False)
    # blue_spectrum.grid(True)
    # green_spectrum.grid(True)
    # red_spectrum.grid(True)
    # plt.show()
