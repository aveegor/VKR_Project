import numpy as np
import cv2.cv2 as cv
from numba import jit, njit
import json
import time


def draw_borders(local_rectangles_img, coordinates_of_patches):

    for cnt, patch in enumerate(coordinates_of_patches):
        start_point = patch[0]
        end_point = patch[1]
        local_rectangles_img = cv.rectangle(local_rectangles_img, start_point, end_point,
                                            (0, 255, 0), 1)
        local_rectangles_img = cv.putText(local_rectangles_img, str(cnt),
                                          (start_point[0], end_point[1] - ((end_point[1] - start_point[1]) // 2)),
                                          cv.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv.LINE_AA)
    return local_rectangles_img


def find_patches(img, size_patch):
    coordinates_of_patches = []
    img_size = [img.shape[1], img.shape[0]]
    local_patches = []
    cnt_patches = 0
    for end_of_row in range(size_patch[1], img_size[1] + size_patch[1], size_patch[1]):
        for end_of_column in range(size_patch[0], img_size[0] + size_patch[0], size_patch[0]):
            start_of_row = end_of_row - size_patch[1]
            start_of_column = end_of_column - size_patch[0]
            local_patches.append(
                img[start_of_row:end_of_row, start_of_column:end_of_column])  # input_img[height, width]

            coordinates_of_patches.append([[start_of_column, start_of_row], [end_of_column, end_of_row]])

            cnt_patches += 1

    return coordinates_of_patches, np.asarray(local_patches)


@njit()
def find_spectrum(img, normalization_flag=True):
    spectrum = np.zeros((3, 256))
    for row_pix in range(img.shape[0]):
        for column_pix in range(img.shape[1]):
            for brightness_level in range(256):
                if img[row_pix][column_pix][0] == brightness_level:
                    spectrum[0][brightness_level] += 1  # Blue channel
                if img[row_pix][column_pix][1] == brightness_level:
                    spectrum[1][brightness_level] += 1  # Green channel
                if img[row_pix][column_pix][2] == brightness_level:
                    spectrum[2][brightness_level] += 1  # Red channel
    if normalization_flag:
        number_of_pixels = img.shape[0] * img.shape[1]
        spectrum = spectrum / number_of_pixels
    return spectrum


@njit()
def find_corrosion(local_patches, samples):
    patch_numbers = np.empty(0)
    cnt = 0
    for patch in local_patches:
        for sample_number in range(len(samples)):
            deviation = (find_spectrum(patch) - samples[sample_number]) ** 2
            mean_deviation = np.sqrt(deviation.mean())
            if mean_deviation <= target_deviation:
                patch_numbers = np.append(patch_numbers, cnt)
                break
        cnt += 1
    return patch_numbers


# settings
size_of_patch = [20, 20]  # [width, height]
size_of_img = [1920, 1080]
bins = 125
target_deviation = 0.0084

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
    input_img = cv.imread('corrosion_10.jpg')
    input_img = cv.resize(input_img, (size_of_img[0], size_of_img[1]), interpolation=cv.INTER_AREA)
    input_img = cv.GaussianBlur(input_img, [3, 3], 0)
    start = time.perf_counter()
    coord_patches, patches = find_patches(input_img, size_of_patch)

    patches_corrosion = find_corrosion(patches, spectrum_samples_val)

    patches_corrosion = patches_corrosion.tolist()

    patches_with_corrosion = []
    for patch_index in patches_corrosion:
        patches_with_corrosion.append(coord_patches[int(patch_index)])
    end = time.perf_counter()
    print(end - start)

    rectangles_img = draw_borders(input_img, patches_with_corrosion)
    cv.imshow('img', rectangles_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

