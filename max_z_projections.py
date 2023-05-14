import argparse
import os
from datetime import datetime

import cv2
import numpy as np


# only use if you want to skip layers in between. otherwise no need
def read_images_in_range(current_path, start, end, steps):
    # images need to be loaded in the correct order so that the range works as expected
    image_numbers = [int(float(os.path.splitext(img_filename)[0])) for img_filename in os.listdir(current_path)]
    image_numbers.sort()

    images = []
    for image_number in image_numbers:
        if start <= image_number <= end and image_number % steps == 0:
            images.append(cv2.imread(f'{current_path}/{image_number}.000.tif', cv2.IMREAD_GRAYSCALE))

    return images


def main():
    # select folders (time points)
    for folder_number in range(START_TIMEPOINT, END_TIMEPOINT + 1):
        start = datetime.now()

        folder_name = f'{folder_number:0>{ZEROS_PADDING_WIDTH}}'
        current_path = f'{PATH_TO_RECONSTRUCTIONS}/{folder_name}'

        images = read_images_in_range(current_path, Z_START_PLANE, Z_END_PLANE, Z_PLANE_JUMP_STEPS)
        # images = [cv2.imread(f'{current_path}/{img_filename}', cv2.IMREAD_GRAYSCALE)

        max_z_projection = np.max(images, axis=0)

        # write the images
        image_name = f'max_z_projection_{DATASET}_zs_{Z_START_PLANE}_ze_{Z_END_PLANE}_{Z_PLANE_JUMP_STEPS}_{folder_name}.tif'
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        cv2.imwrite(f'{OUTPUT_DIR}/{image_name}', max_z_projection)

        end = datetime.now()
        print(end - start, folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Calculate the maximum projection through the Z stack of 
                       reconstructed holograms for the given range of timepoints. 
                       For a detailed description of the parameters, please refer to the README file.''')

    parser.add_argument(
        '--reconstructions', '-r', required=True, type=str,
        help='Path to reconstructions of the holograms - contains folders of all timepoints')

    parser.add_argument(
        '--dataset', '-d', type=str, required=True,
        help='name of the dataset of the recorded experiment - used only for naming the resulting files.')

    parser.add_argument(
        '--output-dir', '-o', type=str,
        help='the directory to which the results will be stored (default: ./<dataset>_max_z_projection_results)')

    parser.add_argument('--z-start-plane', '-zs', required=True, type=int, help='Z-plane start value')

    parser.add_argument('--z-end-plane', '-ze', required=True, type=int, help='Z-plane end value')

    parser.add_argument(
        '--end-timepoint', '-te', type=int, required=True,
        help='a number representing the end timepoint of the range to be considered')

    parser.add_argument(
        '--start-timepoint', '-ts', type=int, default=1,
        help='a number representing the start timepoint of the range to be considered (default: 1)')

    parser.add_argument('--z-plane-jump-steps', '-zj', type=int, default=1, help='Z-plane jump steps (default: 1)')

    parser.add_argument(
        '--zeros-padding-width', type=int, default=5,
        help='Zeros padding width of folder names of reconstructed timepoints (default: 5)')

    args = parser.parse_args()
    DATASET = args.dataset
    PATH_TO_RECONSTRUCTIONS = args.reconstructions  # 'E:/some_data_set/reconstructions/Amplitude'
    Z_START_PLANE = args.z_start_plane  # -170
    Z_END_PLANE = args.z_end_plane  # 26
    Z_PLANE_JUMP_STEPS = args.z_plane_jump_steps  # 1
    START_TIMEPOINT = args.start_timepoint  # 1
    END_TIMEPOINT = args.end_timepoint  # 301
    ZEROS_PADDING_WIDTH = args.zeros_padding_width  # 5
    OUTPUT_DIR = args.output_dir
    if OUTPUT_DIR is None:
        OUTPUT_DIR = f'./{DATASET}_max_z_projection_results'

    main()
