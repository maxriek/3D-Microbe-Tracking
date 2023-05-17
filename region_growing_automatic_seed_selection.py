import argparse
import os

import cv2
import numpy as np

import image_utils
from region_growing import RegionGrowing


def main():
    mhi_npy = np.load(MHI_NPY_FILE)
    detected_blobs = np.load(BLOBS_FILE)

    # resize x and y dimensions to fit the mhi
    detected_blobs[:, :2] = (detected_blobs[:, :2] / 2.0)
    # resize bobs size #TODO: check if relevant
    detected_blobs[:, 3] = (detected_blobs[:, 3] / 2.0)

    filtered_seeds = detected_blobs

    rg = RegionGrowing(threshold=THRESHOLD,
                       radius=RADIUS,
                       rejection_threshold=REJECTION_THRESHOLD,
                       n_values_to_ignore=N_VALUES_TO_IGNORE)

    index = 0
    while len(filtered_seeds) > 0:
        seed = (filtered_seeds[0, :2]).astype(int)
        print('index: ', index)
        print('seed: ', seed)
        print('filtered_seeds.shape: ', filtered_seeds.shape)

        filtered_seeds = filtered_seeds[1:]
        track = rg.region_growing(mhi_npy, seed)
        if track.size < MINIMUM_TRACK_LENGTH:
            print(f'no tracks found or track is too short - length {track.size}')
            continue

        track_img_resized = cv2.resize(track, (2048, 2048), interpolation=cv2.INTER_CUBIC)
        save_results(track_img_resized, seed, index)

        dilation_kernel_size = 30
        dilation_img = image_utils.dilate_img(track, dilation_kernel_size)
        filtered_seeds = filter_seed_points(filtered_seeds, dilation_img)

        index += 1
        print('############################ ###')


def filter_seed_points(detected_centers, track_image):
    # TODO implement with numpy to improve performance
    # filter out centers which don't lie on any visible path on MHI
    filtered_centers = []
    for center in detected_centers.astype(int):
        curr_x = center[0]
        curr_y = center[1]

        if not (track_image[curr_y, curr_x] > 0):
            filtered_centers.append(center)

    return np.array(filtered_centers)


def save_results(track_img, seed, index):
    results_path = f'{OUTPUT_DIR}/tracks_threshold_{THRESHOLD}_radius_{RADIUS}'
    if not os.path.exists(results_path):
        os.makedirs(f'{results_path}/data')

    file_name = f'track_index_{index}_seed_{seed}'
    cv2.imwrite(f'{results_path}/{file_name}.tiff', track_img)
    np.save(f'{results_path}/data/{file_name}.tiff', track_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Perform region growing on a given MHI from a set of given seed point. 
                     For a detailed description of the parameters, please refer to the README file.''')

    parser.add_argument('--mhi-npy-file', '-mhi',
                        type=str,
                        required=True,
                        help='path to the motion history image (MHI) npy-file.')

    parser.add_argument('--blobs-file', '-b',
                        type=str,
                        required=True,
                        help='path to the detected blobs npy-file. '
                             'The file is the result of blob_detection. '
                             'The file contains the blobs of all timepoints regardless of any track segment')

    parser.add_argument('--output-dir', '-o',
                        type=str,
                        help='the directory to which the results will be stored '
                             '(default: ./<dataset>_region_growing_automatic_results)')

    parser.add_argument('--dataset', '-d',
                        type=str,
                        required=True,
                        help='name of the dataset of the MHI - used only for naming the resulting track segment.')

    parser.add_argument('--radius', '-r',
                        type=int,
                        help='search radius for neighboring pixels (default: 5).',
                        default=5)

    parser.add_argument('--threshold',
                        type=int,
                        help='the time difference threshold for comparing two pixels (default: 5).',
                        default=5)

    parser.add_argument('--rejection-threshold',
                        type=int,
                        help='number of rejection votes for a pixel to be excluded from a region (default: 1).',
                        default=1)

    parser.add_argument('--n-values-to-ignore', '-n',
                        type=int,
                        help='used to filter out the most prominent n values in the mhi (default: 3).',
                        default=3)

    parser.add_argument('--min-track-length',
                        type=int,
                        help='tracks smaller than the minimum track length (in pixel count) will be ignored '
                             '(default: 500).',
                        default=500)

    args = parser.parse_args()

    DATASET = args.dataset
    MHI_NPY_FILE = args.mhi_npy_file
    BLOBS_FILE = args.blobs_file
    RADIUS = args.radius
    THRESHOLD = args.threshold
    REJECTION_THRESHOLD = args.rejection_threshold
    N_VALUES_TO_IGNORE = args.n_values_to_ignore
    MINIMUM_TRACK_LENGTH = args.min_track_length

    OUTPUT_DIR = args.output_dir
    if OUTPUT_DIR is None:
        OUTPUT_DIR = f'./{DATASET}_region_growing_automatic_results'

    main()
