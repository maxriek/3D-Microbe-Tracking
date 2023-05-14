import os

import cv2 as cv
import numpy as np


def get_time_string_from_max_z_file_name(max_z_file_name):
    if max_z_file_name[-9:-4].isdigit():
        return max_z_file_name[-9:-4]
    else:
        raise ValueError("max-z projection file name is malformed")


def load_z_layers_at_time_point(path, min_file_nr, max_file_nr):
    images = []
    list_of_files = os.listdir(path)
    list_of_files = list(map(lambda x: x[:-4], list_of_files))
    list_of_files.sort(key=float)
    list_of_files = list(map(lambda x: f'{x}.tif', list_of_files))

    for img_file in list_of_files:
        file_nr = int(float((os.path.splitext(img_file)[0])))
        if min_file_nr <= file_nr <= max_file_nr:
            images.append(cv.imread(f'{path}/{file_nr}.000.tif', cv.IMREAD_GRAYSCALE))

    return np.array(images)
