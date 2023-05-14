import cv2
import numpy as np


def get_x_y(point):
    point_x = int(float(point[0]))
    point_y = int(float(point[1]))
    return point_x, point_y


def get_region_boundaries(image, target_pixel, radius):
    cx, cy = get_x_y(target_pixel)

    cx_start = cx - radius
    if cx_start < 0:
        cx_start = 0

    cx_end = cx + radius
    if cx_end > image.shape[1] - 1:
        cx_end = image.shape[1] - 1

    cy_start = cy - radius
    if cy_start < 0:
        cy_start = 0

    cy_end = cy + radius
    if cy_end > image.shape[0] - 1:
        cy_end = image.shape[0] - 1

    return [cx_start, cx_end, cy_start, cy_end]


def dilate_img(img, kernel_size):
    dilation_kernel_size = kernel_size
    kernel_dilation = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    kernel_erosion = np.ones((1, 1), np.uint8)
    erosion_img = cv2.erode(img, kernel_erosion, iterations=1)
    dilation_img = cv2.dilate(erosion_img, kernel_dilation, iterations=1)
    return dilation_img

def fill_img(img, kernel_size):
    structuring_kernel_size = kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (structuring_kernel_size, structuring_kernel_size))
    filled_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return filled_img