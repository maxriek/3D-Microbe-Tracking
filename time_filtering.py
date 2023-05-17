import argparse

import cv2
import numpy as np

import image_utils


def main():
    track_2d = np.load(TRACK_2D_NPY_FILE_PATH)
    track_3d_coordinates = np.load(TRACK_3D_NPY_FILE_PATH)
    mhi_ = np.load(MHI_NPY_FILE_PATH)  # load MHI
    mhi = cv2.resize(mhi_.astype(float), (2048, 2048), interpolation=cv2.INTER_CUBIC)

    # filter centers which belong to path segment
    blobs_on_segment = []
    for blob in track_3d_coordinates:
        center_x, center_y = image_utils.get_x_y(blob)
        if track_2d[center_y, center_x] > 0 and in_time_tolerance_range(blob, mhi):
            blobs_on_segment.append(blob)

    np.save(f'{TRACK_3D_NPY_FILE_PATH[:-4]}__{MHI_TIME_TOLERANCE}.npy', blobs_on_segment)
    return

def in_time_tolerance_range(blob, mhi):
    x, y = image_utils.get_x_y(blob)
    blob_time_point = int(float(blob[3]))

    time_on_mhi = mhi[y, x]

    return time_on_mhi - MHI_TIME_TOLERANCE <= blob_time_point <= time_on_mhi + MHI_TIME_TOLERANCE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script eliminates unrelated particles from the 3D track by comparing the time associated '
                    'with each particle to the corresponding time value on the MHI.')

    parser.add_argument('--mhi-npy-file', '-mhi',
                        type=str,
                        required=True,
                        help='path to the motion history image (MHI) npy-file.')

    parser.add_argument('--track-2d', '-ts2d',
                        type=str,
                        required=True,
                        help='path to the 2D track segment npy-file, which corresponds to the 3D track')

    parser.add_argument('--track-3d', '-ts3d',
                        type=str,
                        required=True,
                        help='path to the 3D track segment npy-file, which will be filtered')

    parser.add_argument('--mhi-time-tolerance',
                        type=int,
                        help='MHI time tolerance - if not provided no filtering will take place.',
                        default=None)

    args = parser.parse_args()

    MHI_NPY_FILE_PATH = args.mhi_npy_file
    TRACK_2D_NPY_FILE_PATH = args.track_2d
    TRACK_3D_NPY_FILE_PATH = args.track_3d
    MHI_TIME_TOLERANCE = args.mhi_time_tolerance

    main()
