import argparse
import os
from datetime import datetime

import cv2 as cv
import numpy as np

import file_utils
import find_focus as ff
import image_utils


def in_time_tolerance_range(point, mhi):
    if MHI_TIME_TOLERANCE is None:
        return True

    center_x, center_y = image_utils.get_x_y(point)
    center_time_point = int(float(point[2]))
    time_on_mhi = mhi[center_y, center_x]

    return time_on_mhi - MHI_TIME_TOLERANCE <= center_time_point <= time_on_mhi + MHI_TIME_TOLERANCE


def get_path_to_time_point(current_loaded_max_z):
    # time_folder_name = get_time_string_from_max_z_file_name(current_loaded_max_z)
    return f'{PATH_TO_RECONSTRUCTIONS}/{int(current_loaded_max_z):0>{ZEROS_PADDING_WIDTH}}'


def get_blobs_on_segment(max_z_blobs, track_segment, mhi):
    blobs_on_track_segment = []
    for blob in max_z_blobs:
        blob_x, blob_y = image_utils.get_x_y(blob)
        if track_segment[blob_y, blob_x] > 0 and in_time_tolerance_range(blob, mhi):
            blobs_on_track_segment.append(blob)

    return blobs_on_track_segment


def save_blobs_on_track_segment(blobs_on_track_segment, track_name):
    if not os.path.exists(BLOBS_ON_2D_TRACK_OUTPUT_DIR):
        os.makedirs(BLOBS_ON_2D_TRACK_OUTPUT_DIR)

    blobs_on_track_segment_filename = f'{BLOBS_ON_2D_TRACK_OUTPUT_DIR}/blobs_on_track_segment_{track_name}'
    np.save(blobs_on_track_segment_filename, blobs_on_track_segment)


def build_3d_coordinate_with_time(blob, z):
    blob_x, blob_y = image_utils.get_x_y(blob)
    blob_diameter = float(blob[3])
    max_z_time_point = blob[2]
    return [blob_x, blob_y, z, max_z_time_point, blob_diameter]


def find_z_coordinates_of_blobs(blobs_on_track_segment):
    blobs_3d_coordinates_tg_small = []
    blobs_3d_coordinates_tg_large = []
    blobs_3d_coordinates_ala_max_small = []
    blobs_3d_coordinates_ala_max_large = []

    current_loaded_max_z = blobs_on_track_segment[0][2]
    path_to_time_point = get_path_to_time_point(current_loaded_max_z)

    all_z_layers_at_current_time = file_utils.load_z_layers_at_time_point(path_to_time_point, Z_START_PLANE, Z_END_PLANE)

    start = datetime.now()
    for blob in blobs_on_track_segment:
        blob_diameter = int(float(blob[3]))

        print(f'progress on blob: {blob}')
        print(f'blob_diameter: {blob[3]}')
        if blob_diameter < MIN_DIAMETER_SIZE:
            print(f'blob skipped..')
            continue

        max_z_time_point = blob[2]
        if current_loaded_max_z != max_z_time_point:
            current_loaded_max_z = max_z_time_point
            path_to_time_point = get_path_to_time_point(current_loaded_max_z)
            all_z_layers_at_current_time = file_utils.load_z_layers_at_time_point(path_to_time_point, Z_START_PLANE, Z_END_PLANE)

        z_coordinate_tenengrad_small = ff.find_focus_layer_tenengrad_focus_metric(blob,
                                                                                  all_z_layers_at_current_time,
                                                                                  radius=int(blob_diameter / 2))
        z_coordinate_tenengrad_large = ff.find_focus_layer_tenengrad_focus_metric(blob,
                                                                                  all_z_layers_at_current_time,
                                                                                  radius=blob_diameter)

        z_coordinate_ala_max_small = ff.find_focus_layer_ala_max(blob,
                                                                 all_z_layers_at_current_time,
                                                                 radius=int(blob_diameter / 2))
        z_coordinate_ala_max_large = ff.find_focus_layer_ala_max(blob,
                                                                 all_z_layers_at_current_time,
                                                                 radius=blob_diameter)

        blobs_3d_coordinates_tg_small.append(build_3d_coordinate_with_time(blob, z_coordinate_tenengrad_small))
        blobs_3d_coordinates_tg_large.append(build_3d_coordinate_with_time(blob, z_coordinate_tenengrad_large))
        blobs_3d_coordinates_ala_max_small.append(build_3d_coordinate_with_time(blob, z_coordinate_ala_max_small))
        blobs_3d_coordinates_ala_max_large.append(build_3d_coordinate_with_time(blob, z_coordinate_ala_max_large))

    print('time total: ', datetime.now() - start)

    return blobs_3d_coordinates_tg_small, blobs_3d_coordinates_tg_large, blobs_3d_coordinates_ala_max_small, blobs_3d_coordinates_ala_max_large


def main():
    max_z_blobs = np.load(BLOBS_FILE)
    mhi_ = np.load(MHI_NPY_FILE)
    mhi = cv.resize(mhi_.astype(float), (2048, 2048), interpolation=cv.INTER_CUBIC)

    for track_name in os.listdir(PATH_TO_TRACK_SEGMENTS_2D):
        track_segment = np.load(f'{PATH_TO_TRACK_SEGMENTS_2D}/{track_name}')
        if track_segment.shape == (2048, 2048, 3):
            track_segment = track_segment[:, :, 0]

        # filter centers which belong to path_to_time_point segment
        blobs_on_track_segment = get_blobs_on_segment(max_z_blobs, track_segment, mhi)
        save_blobs_on_track_segment(blobs_on_track_segment, track_name)

        print('finding focus for all microbes (centers)...')

        blobs_3d_coordinates_tg_small, blobs_3d_coordinates_tg_large, blobs_3d_coordinates_ala_max_small, blobs_3d_coordinates_ala_max_large = find_z_coordinates_of_blobs(
            blobs_on_track_segment)

        track_3d_file_path = f'{OUTPUT_DIR}/3d_{track_name}'
        np.save(f'{track_3d_file_path}_ala_tenengrad_small', blobs_3d_coordinates_tg_small)
        np.save(f'{track_3d_file_path}_ala_tenengrad_large', blobs_3d_coordinates_tg_large)
        np.save(f'{track_3d_file_path}_ala_max_small', blobs_3d_coordinates_ala_max_small)
        np.save(f'{track_3d_file_path}_ala_max_large', blobs_3d_coordinates_ala_max_large)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Find the best estimate of the z coordinate of each blob on a given track segment.
                     For a detailed description of the usage and parameters, please refer to the README file.''')

    # Add arguments
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='name of the dataset of the MHI')

    parser.add_argument('--mhi-npy-file', '-mhi', type=str, required=True,
                        help='path to the motion history image (MHI) npy-file.')

    parser.add_argument('--reconstructions', '-r', type=str, required=True,
                        help='Path to reconstructions of the holograms - contains folders of all timepoints')

    parser.add_argument('--tracks', '-tr', type=str, required=True,
                        help='Path to the 2D track segments, which are the results of region growing')

    parser.add_argument('--blobs-file', '-b', type=str, required=True,
                        help='path to the detected blobs npy-file. '
                             'The file is the result of blob_detection. '
                             'The file contains the blobs of all timepoints regardless of any track segment')

    parser.add_argument('--z-start-plane', '-zs', type=int, required=True,
                        help='Z-plane start value - '
                             'this should be the exact value used to calculate the may-Z projections')

    parser.add_argument('--z-end-plane', '-ze',type=int, required=True,
                        help='Z-plane end value - '
                             'this should be the exact value used to calculate the may-Z projections')

    parser.add_argument('--output-dir', '-o', type=str,
                        help='the directory to which the results will be stored '
                             '(default: ./<dataset>_z_layer_selection_results)')

    parser.add_argument('--mhi-time-tolerance', type=int, default=None,
                        help='MHI time tolerance - if not provided no filtering will take place. '
                             'Please refer to the README for more information')

    parser.add_argument('--min-diameter-size', type=int, default=0,
                        help='Blobs with smaller diameter size will not be considered (default: 0)')

    parser.add_argument('--zeros-padding-width', type=int, default=5,
                        help='Zeros padding width of folder names of reconstructed timepoints (default: 5)')

    # Parse arguments
    args = parser.parse_args()

    # Assign the values to variables
    DATASET = args.dataset
    MHI_NPY_FILE = args.mhi_npy_file
    PATH_TO_RECONSTRUCTIONS = args.reconstructions
    PATH_TO_TRACK_SEGMENTS_2D = args.tracks

    BLOBS_FILE = args.blobs_file
    MHI_TIME_TOLERANCE = args.mhi_time_tolerance
    Z_START_PLANE = args.z_start_plane
    Z_END_PLANE = args.z_end_plane
    MIN_DIAMETER_SIZE = args.min_diameter_size
    ZEROS_PADDING_WIDTH = args.zeros_padding_width

    OUTPUT_DIR = args.output_dir
    if OUTPUT_DIR is None:
        OUTPUT_DIR = f'./{DATASET}_z_layer_selection_results'

    # --- intermediary files - only relevant for analysis
    BLOBS_ON_2D_TRACK_OUTPUT_DIR = f'{OUTPUT_DIR}/blobs_on_track_segment/'

    main()
