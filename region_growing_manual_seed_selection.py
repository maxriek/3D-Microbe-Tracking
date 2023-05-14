import argparse
import os
from datetime import datetime

import cv2
import numpy as np

import image_utils
from region_growing import RegionGrowing

# store the mouse clicks on mouse_callback
clicks = []
## ---
MAIN_WINDOW_NAME = 'region_growing'


def main():
    mhi = np.load(f'{PATH_TO_MHI}/{DATASET}_mhi.npy')
    img = cv2.imread(f'{PATH_TO_MHI}/{DATASET}_mhi.png', 1)

    cv2.namedWindow(MAIN_WINDOW_NAME)
    cv2.setMouseCallback(MAIN_WINDOW_NAME, on_mouse)
    cv2.imshow(MAIN_WINDOW_NAME, img)
    cv2.waitKey()

    start = datetime.now()
    print("start time: ", start)

    rg = RegionGrowing(threshold=THRESHOLD,
                       radius=RADIUS,
                       rejection_threshold=REJECTION_THRESHOLD,
                       n_values_to_ignore=N_VALUES_TO_IGNORE,
                       progress_callback=custom_progress)

    seed = get_seed()
    log_seed(seed, mhi)
    track_img = rg.region_growing(mhi, seed)

    end = datetime.now()
    params = create_params(end - start)
    post_process_and_save_results(track_img, params)

    print("end time: ", end)
    print("time elapsed: ", end - start)

    # make sure to call the show progress method one last time (for short tracks < 1000 iterations)
    custom_progress(track_img)

    cv2.waitKey()
    cv2.destroyAllWindows()

def on_mouse(event, x, y, flags, params):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append([x, y])


def get_seed():
    global SEED_X, SEED_Y
    if SEED_X is not None and SEED_Y is not None:
        seed = [SEED_X, SEED_Y]
    else:
        seed = clicks[-1]
        [SEED_X, SEED_Y] = seed

    return seed


def log_seed(seed, mhi):
    [x, y] = seed
    print(f'seed: {seed}')
    print(f'Seed value on mhi: {mhi[y, x]}')


def post_process_and_save_results(track_img, params):
    padded_track_folder_name = f'{TRACK_NR:0>{3}}'
    results_path = os.path.join(OUTPUT_DIR, padded_track_folder_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    PARAMS_AS_STRING = f'Track_{TRACK_NR}_Radius_{RADIUS}_Threshold_{THRESHOLD}_RJThrshld_{REJECTION_THRESHOLD}_VALUES_TO_IGNORE{N_VALUES_TO_IGNORE}_SEED_X{SEED_X}_SEED_Y{SEED_Y}'
    base_name = f'{DATASET}_{PARAMS_AS_STRING}'

    track_img_resized = cv2.resize(track_img, (2048, 2048), interpolation=cv2.INTER_CUBIC)
    filepath = os.path.join(results_path, base_name)
    np.save(filepath, track_img_resized)
    cv2.imwrite(f'{filepath}.tiff', track_img_resized)

    structuring_kernel_size = 30
    filled_img = image_utils.fill_img(track_img_resized, structuring_kernel_size)
    filename = f'{base_name}_filled_image_kernel_{structuring_kernel_size}'
    filepath = os.path.join(results_path, filename)
    np.save(filepath, filled_img)
    cv2.imwrite(f'{filepath}.tif', filled_img)

    dilation_kernel_size = 30
    dilation_img = image_utils.dilate_img(track_img_resized, dilation_kernel_size)
    filename = f'{base_name}_dilated_kernel_{dilation_kernel_size}'
    filepath = os.path.join(results_path, filename)
    np.save(filepath, dilation_img)
    cv2.imwrite(f'{filepath}.tif', dilation_img)

    with open(os.path.join(results_path, f'region_growing_params_{base_name}.txt'), mode='w') as file:
        for key, value in params.items():
            file.write(f'{key}: {value}\n')


def create_params(elapsed_time):
    params = {
        "DATASET": DATASET,
        "TRACK_NR": TRACK_NR,
        "RADIUS": RADIUS,
        "THRESHOLD": THRESHOLD,
        "REJECTION_THRESHOLD": REJECTION_THRESHOLD,
        "N_VALUES_TO_IGNORE": N_VALUES_TO_IGNORE,
        "SEED_X": SEED_X,
        "SEED_Y": SEED_Y,
        "elapsed_time": elapsed_time
    }
    return params


def custom_progress(track_img):
    cv2.imshow(MAIN_WINDOW_NAME, track_img)
    cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Perform region growing on a given MHI from a manually selected seed point. 
                     For a detailed description of the parameters, please refer to the README file.''')

    parser.add_argument(
        '--path-to-mhi', '-mhi', type=str, required=True,
        help='path to the motion history image (MHI) file.')

    parser.add_argument(
        '--output-dir', '-o', type=str,
        help='the directory to which the results will be stored (default: ./<dataset>_region_growing_results)')

    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='name of the dataset of the MHI - used only for naming the resulting track segment.')

    parser.add_argument('--track-nr', '-t', type=int, default=1,
                        help='track number - used only for naming the resulting track segment (default: 1).')

    parser.add_argument('--radius', '-r', type=int, default=5,
                        help='search radius for neighboring pixels (default: 5).')

    parser.add_argument('--threshold', type=int, default=5,
                        help='the time difference threshold for comparing two pixels (default: 5).')

    parser.add_argument('--rejection-threshold', type=int, default=1,
                        help='number of rejection votes for a pixel to be excluded from a region (default: 1).')

    parser.add_argument('--n-values-to-ignore', '-n', type=int, default=3,
                        help='used to filter out the most prominent n values in the mhi (default: 3).')

    parser.add_argument('--seed-x', '-x', type=int, default=None,
                        help='x-coordinate of the seed point for the region growing - only used if provided along with SEED_Y.')

    parser.add_argument('--seed-y', '-y', type=int, default=None,
                        help='y-coordinate of the seed point for the region growing - only used if provided along with SEED_X.')

    args = parser.parse_args()

    DATASET = args.dataset
    PATH_TO_MHI = args.path_to_mhi
    TRACK_NR = args.track_nr
    RADIUS = args.radius
    THRESHOLD = args.threshold
    REJECTION_THRESHOLD = args.rejection_threshold
    N_VALUES_TO_IGNORE = args.n_values_to_ignore
    SEED_X = args.seed_x
    SEED_Y = args.seed_y
    OUTPUT_DIR = args.output_dir
    if OUTPUT_DIR is None:
        OUTPUT_DIR = f'./{DATASET}_region_growing_results'

    main()
