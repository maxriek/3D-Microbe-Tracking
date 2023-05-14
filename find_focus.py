import cv2
import numpy as np

import image_utils


def find_focus_layer_ala_max(blob,
                             all_z_layers_at_current_center,
                             radius,
                             min_brightness_threshold=80,
                             max_brightness_threshold=100):
    [cx_start, cx_end, cy_start, cy_end] = image_utils.get_region_boundaries(all_z_layers_at_current_center[0],
                                                                             blob,
                                                                             radius)
    cropped_layers = all_z_layers_at_current_center[:, cy_start:cy_end, cx_start:cx_end]
    # boundaries for 10 pixels radius
    # 70,70 --- 70,90
    #   |   ---   |
    # 90,70 --- 90,90

    means_bright = [0] * cropped_layers.shape[0]
    # variance_dark = [0]*cropped_regions.shape[0]
    index = 0
    for z in cropped_layers:
        z_raveled = z.ravel()
        bright_filter = filter(lambda p: (p > min_brightness_threshold), z_raveled)
        bright_pixels = list(bright_filter)
        means_bright[index] = np.mean(bright_pixels) if bright_pixels else 0

        # dark_filter = filter(lambda p: (p < max_brightness_threshold), z_raveled)
        # dark_pixels = list(dark_filter)
        # variance_dark[index] = np.var(dark_pixels) if dark_pixels else 255

        index += 1

    max_mean = np.max(means_bright)
    # min_variance = np.min(variance_dark)

    return means_bright.index(max_mean)


def compute_tenengrad_focus_metric(plane):
    gx = cv2.Sobel(plane, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(plane, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(gx ** 2 + gy ** 2)


def find_focus_layer_tenengrad_focus_metric(blob, z_layers, radius=0):
    cropped_regions = z_layers
    if radius != 0:
        [cx_start, cx_end, cy_start, cy_end] = image_utils.get_region_boundaries(z_layers[0], blob, radius)
        cropped_regions = z_layers[:, cy_start:cy_end, cx_start:cx_end]

    focus_metric_list = []
    for cropped_z_layer in cropped_regions:
        focus_metric_list.append(compute_tenengrad_focus_metric(cropped_z_layer))

    return np.argmax(focus_metric_list)
