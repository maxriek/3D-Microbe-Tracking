import cv2
import numpy as np

import image_utils


MHI_TIME_TOLERANCE = 999  # 305 #set filter time tolerane


def in_time_tolerance_range(point, mhi):
    center_x, center_y = image_utils.get_x_y(point)
    center_time_point = int(float(point[3]))

    time_on_mhi = mhi[center_y, center_x]

    return time_on_mhi - MHI_TIME_TOLERANCE <= center_time_point <= time_on_mhi + MHI_TIME_TOLERANCE


def main():
    track_segment_filename = \
        './' #set track segment filename
    track_3d_coordinates_filename = \
        './' #set 3d_coordinates  filename

    track_segment = np.load(f'{track_segment_filename}.npy')
    track_3d_coordinates = np.load(f'{track_3d_coordinates_filename}.npy')
    mhi_ = np.load('./') #load MHI
    mhi = cv2.resize(mhi_.astype(float), (2048, 2048), interpolation=cv2.INTER_CUBIC)

    # filter centers which belong to path segment
    blobs_on_segment = []
    for point in track_3d_coordinates:
        center_x, center_y = image_utils.get_x_y(point)
        if (track_segment[center_y, center_x] > 0).any() and in_time_tolerance_range(point, mhi):
            blobs_on_segment.append(point)

    np.save(f'{track_3d_coordinates_filename}__{MHI_TIME_TOLERANCE}.npy', blobs_on_segment)
    return


if __name__ == '__main__':
    main()
