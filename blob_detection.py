import argparse
import os
from datetime import datetime

import cv2 as cv
import numpy as np

import file_utils


def detect_blobs(im):
    # invert image colors because simpleBlobDetector seems to expect bright background and dark blobs
    im = cv.bitwise_not(im)
    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = MAX_THRESHOLD

    # Filter by Area.
    params.filterByArea = True
    params.minArea = MIN_AREA
    params.maxArea = MAX_AREA
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0

    # Create a detector with the parameters
    detector = cv.SimpleBlobDetector_create(params)

    return detector.detect(im)


def save_results(detected_blobs):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    blobs_filename = f'{DATASET}_Blobs_params_maxThreshold_{MAX_THRESHOLD}_minArea{MIN_AREA}_maxArea{MAX_AREA}'
    np.save(os.path.join(OUTPUT_DIR, blobs_filename), detected_blobs)

    params = {
        "DATASET": DATASET,
        "MAX_THRESHOLD": MAX_THRESHOLD,
        "MIN_AREA": MIN_AREA,
        "MAX_AREA": MAX_AREA
    }

    with open(os.path.join(OUTPUT_DIR, f'{DATASET}_blobs_detection_params.txt'), mode='w') as file:
        for key, value in params.items():
            file.write(f'{key}: {value}\n')


def main():
    script_start = datetime.now()
    print("start time: ", script_start)

    detected_blobs = []
    list_of_images = os.listdir(MAX_Z_BASE_PATH)
    list_of_images.sort()
    for image_filename in list_of_images:
        start = datetime.now()
        print(f'processing: {image_filename} ## ')

        img = cv.imread(os.path.join(MAX_Z_BASE_PATH, image_filename))
        key_points = detect_blobs(img)

        for c in key_points:
            time = int(file_utils.get_time_string_from_max_z_file_name(image_filename))
            detected_blobs.append([c.pt[0], c.pt[1], time, c.size])

        save_results(detected_blobs)
        end = datetime.now()
        print("elapsed time: ", end - start)

    save_results(detected_blobs)
    script_end = datetime.now()
    print("end time: ", script_end)
    print("total time elapsed: ", script_end - script_start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Apply OpenCV SimpleBlobDetector on provided Max-Z projection images and save the results in 
        an npy-file. For a detailed description of the parameters, please refer to the README file.''')

    parser.add_argument('--max-z-projections', type=str, required=True,
                        help='path to the Max-Z projection files of all time points.')

    parser.add_argument('--output-dir', '-o', type=str,
                        help='path to the directory where the results will be stored (default: ./<dataset>_blob_detection_results)')

    parser.add_argument('--dataset', type=str, required=True,
                        help='name of the dataset of the MHI - used only for naming the resulting track segment.')

    parser.add_argument('--min-area', type=int, default=60,
                        help='change the minimum area of blobs - smaller blobs will be filtered out(default: 60).')

    parser.add_argument('--max-area', type=int, default=300,
                        help='change the maximum area of blobs - smaller blobs will be filtered out(default: 300).')

    parser.add_argument('--max-threshold', type=int, default=255,
                        help='change the maxThreshold value for openCV SimpleBlobDetector (default: 255).')

    args = parser.parse_args()

    MAX_Z_BASE_PATH = args.max_z_projections
    DATASET = args.dataset
    MAX_THRESHOLD = args.max_threshold  # 255
    MIN_AREA = args.min_area  # 60
    MAX_AREA = args.max_area  # 300
    OUTPUT_DIR = args.output_dir
    if OUTPUT_DIR is None:
        OUTPUT_DIR = f'./{DATASET}_blob_detection_results'

    main()
