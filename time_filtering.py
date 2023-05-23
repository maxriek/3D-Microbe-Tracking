import argparse
import os

import cv2
import numpy as np

import image_utils


def main():
    track_2d = np.load(TRACK_2D_NPY_FILE_PATH)

    # older version of the region growing saved the track with 3 channels. Only the 1st is relevant
    if len(track_2d.shape) > 2:
        track_2d = track_2d[: ,: , 0]

    track_3d_coordinates = np.load(TRACK_3D_NPY_FILE_PATH)
    mhi_ = np.load(MHI_NPY_FILE_PATH)  # load MHI
    mhi = cv2.resize(mhi_.astype(float), (2048, 2048), interpolation=cv2.INTER_CUBIC)

    # filter centers which belong to path segment
    blobs_on_segment = []
    for blob in track_3d_coordinates:
        blob_x, blob_y = image_utils.get_x_y(blob)
        if track_2d[blob_y, blob_x] > 0 and in_time_tolerance_range(blob, mhi):
            blobs_on_segment.append(blob)

    np.save(f'{OUTPUT_DIR}/{OUTPUT_BASE_FILENAME}_mhi_time_tolerance_{MHI_TIME_TOLERANCE}.npy', blobs_on_segment)
    return

def in_time_tolerance_range(blob, mhi):
    if MHI_TIME_TOLERANCE is None:
        return True

    x, y = image_utils.get_x_y(blob)
    blob_time_point = int(float(blob[3]))

    time_on_mhi = mhi[y, x]

    return abs(time_on_mhi - MHI_TIME_TOLERANCE) <= blob_time_point <= time_on_mhi + MHI_TIME_TOLERANCE


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

    parser.add_argument('--output-dir', '-o',
                        type=str,
                        help='the directory to which the results will be stored '
                             '(default: ./<dataset>_time_filtering_results)')

    parser.add_argument('--dataset', '-d',
                        type=str,
                        required=True,
                        help='name of the dataset of the MHI - used only for naming the resulting track segment.')

    args = parser.parse_args()

    MHI_NPY_FILE_PATH = args.mhi_npy_file
    TRACK_2D_NPY_FILE_PATH = args.track_2d
    TRACK_3D_NPY_FILE_PATH = args.track_3d
    MHI_TIME_TOLERANCE = args.mhi_time_tolerance
    DATASET = args.dataset

    OUTPUT_DIR = args.output_dir
    if OUTPUT_DIR is None:
        OUTPUT_DIR = f'./{DATASET}_time_filtering_results'

    # --------------------------------------------------------------
    TRACK_3D_NPY_FILE_NAME = os.path.basename(TRACK_3D_NPY_FILE_PATH)
    OUTPUT_BASE_FILENAME = os.path.splitext(TRACK_3D_NPY_FILE_NAME)[0]


    main()
