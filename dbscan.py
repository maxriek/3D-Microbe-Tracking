import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN


# see https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
def main():
    coordinates_3d = np.load(TRACK_3D_NPY_FILE_PATH).astype(float)
    coordinates_3d = convert_to_micro_meters(coordinates_3d)

    time = coordinates_3d[:, 3]
    coordinates_3d_with_time = coordinates_3d[:, :4]

    # -- Plot before dbscan to see how everything looks like --------------------------------------
    axis_3d_no_dbscan = plot_3d_with_color(coordinates_3d_with_time)
    # plot_color_bar_of_timepoints(time)
    # plt.savefig(f'{TRACK_3D_NPY_BASE_FILENAME}.png', format='png')
    # plt.show()
    # -- ------------------------------------------------------------------------------------------

    ## -- this is all it takes for DBSCAN ---------------------------------------------------------
    normalized_centers_with_time = normalize(coordinates_3d_with_time)
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric=custom_distance)
    db.fit(normalized_centers_with_time)
    # -- ------------------------------------------------------------------------------------------

    n_clusters_ = get_num_of_clusters(db)
    n_noise_ = get_num_of_noise_points(db)
    log_n_clusters_and_n_noise(n_clusters_, n_noise_)

    # -- Plot after dbscan - no clouds around clusters - noise points removed ---------------------
    # axis_3d_no_noise = plot_3d_without_noise_no_cloud(coordinates_3d_with_time, db)
    # plt.show()
    # -- ------------------------------------------------------------------------------------------

    # -- Plot after dbscan - plot the clouds around clusters and save results to disk--------------
    plot_3d_with_clouds(coordinates_3d,
                        db,
                        plot_noise=PLOT_NOISE_POINTS,
                        save_clusters=SAVE_CLUSTERS_TO_DISK,
                        axis_3d=axis_3d_no_dbscan)

    if PLOT_NOISE_POINTS:
        plt.savefig(f'{TRACK_3D_NPY_BASE_FILENAME}_clusters_{n_clusters_}_n_noise_points_{n_noise_}.png', format='png')
    else:
        plt.savefig(
            f'{TRACK_3D_NPY_BASE_FILENAME}_clusters_{n_clusters_}_n_noise_points_{n_noise_}_no_noise_points.png',
            format='png')

    plt.show()
    # -- ------------------------------------------------------------------------------------------


def normalize(coordinates_3d_with_time):
    # calculate the norm of each column
    max_values = np.max(coordinates_3d_with_time, axis=0)
    # normalize the columns
    normalized_centers_with_time = coordinates_3d_with_time / max_values

    return normalized_centers_with_time


def weighted_point(p):
    wp = np.array(p[0:4])
    # scale z axis
    wp[2] = 4 * wp[2]

    return wp


def custom_distance(p1, p2):
    dist = np.linalg.norm(weighted_point(p1) - weighted_point(p2))

    return dist


def convert_to_micro_meters(coordinates_3d):
    # convert pixels to micro meter
    coordinates_3d[:, 0:2] = coordinates_3d[:, 0:2] * (330.0 / 2048)  # convert X-Y from pixel to micro meter values
    coordinates_3d[:, 2] = coordinates_3d[:, 2] * 4  # convert z to correct micro meter values

    return coordinates_3d


def create_time_points_colors(time):
    time_colors = plt.cm.rainbow(np.linspace(0, 1, int(max(time)) + 1))
    time_points_colors = [time_colors[int(tp)] for tp in time]
    return time_points_colors


def get_num_of_noise_points(db):
    n_noise_ = list(db.labels_).count(-1)
    return n_noise_


def get_num_of_clusters(db):
    labels = db.labels_
    unique_labels = set(labels)
    # number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)

    return n_clusters_


def create_core_samples_mask(db):
    labels = db.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    return core_samples_mask


def setup_3d_plot():
    # setup 3d plot
    fig = plt.figure(figsize=(12, 8))
    axis = fig.add_subplot(111, projection='3d')
    axis.set_xlim(0, 365)
    axis.set_ylim(0, 365)
    axis.set_zlim(0, 400)
    axis.set_xlabel("x [µm]")
    axis.set_ylabel("y [µm]")
    axis.set_zlabel("z [µm]")
    axis.view_init(elev=40, azim=-166)
    return axis


def plot_3d_with_color(coordinates_3d_with_time):
    time = coordinates_3d_with_time[:, 3]
    time_points_colors = create_time_points_colors(time)

    axis_3d = setup_3d_plot()
    axis_3d.scatter(coordinates_3d_with_time[:, 0]
                    , coordinates_3d_with_time[:, 1]
                    , coordinates_3d_with_time[:, 2]
                    , "o"
                    , s=10
                    , facecolor=time_points_colors
                    , edgecolors=time_points_colors)
    return axis_3d


def plot_color_bar_of_timepoints(time):
    time_points_colors = create_time_points_colors(time)

    custom_cmap = ListedColormap(time_points_colors)
    # create a custom Normalize object to map colors to a range of values
    norm = mpl.colors.Normalize(vmin=0, vmax=len(time_points_colors) - 1)
    # create a ScalarMappable object with the custom colormap and normalization
    sm = ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    # create a color bar using the ScalarMappable object
    cbar = plt.colorbar(sm, orientation='horizontal')
    cbar.set_label('Time index', fontsize=14)
    cbar.ax.tick_params(labelsize=12)


def plot_3d_without_noise_no_cloud(coordinates_3d_with_time, db):
    core_samples_mask = create_core_samples_mask(db)
    labels = db.labels_

    axis_3d = setup_3d_plot()
    mask = labels == -1
    xyzt = coordinates_3d_with_time[~mask & core_samples_mask]
    plot_3d_with_color(xyzt)
    return axis_3d


def plot_3d_with_clouds(coordinates_3d, db, plot_noise, save_clusters, axis_3d=None):
    if axis_3d is None:
        axis_3d = setup_3d_plot()
    coordinates_3d_with_time = coordinates_3d[:, :4]

    labels = db.labels_
    unique_labels = set(labels)
    n_clusters_ = get_num_of_clusters(db)
    n_noise_ = get_num_of_noise_points(db)

    label_colors = [plt.cm.rainbow(each) for each in np.linspace(0, 1, len(unique_labels))]
    core_samples_mask = create_core_samples_mask(db)

    plt.title(f'Estimated number of clusters: {n_clusters_}\n'
              f'Estimated number of noise points: {n_noise_}')

    # create a legend with label colors
    handles = [plt.Rectangle((0, 0), 1, 1, fc=tuple(label_colors[i])) for i in range(len(unique_labels))]
    legend_labels = [f'Track {i + 1}' for i in range(len(unique_labels) - 1)]
    axis_3d.legend(handles, legend_labels, framealpha=0.0)

    for k, col in zip(unique_labels, label_colors):
        # use black for noise points
        if k == -1:
            col = [0, 0, 0, 1]
            if not plot_noise:
                continue

        class_member_mask = labels == k

        # -- comment in if you want to see the class number on each point
        '''
        xyzt = coordinates_3d_with_time[class_member_mask & core_samples_mask]
        x = xyzt[:, 0]
        y = xyzt[:, 1]
        z = xyzt[:, 2]

        for i, txt in enumerate(labels[class_member_mask & core_samples_mask]):
            axis_3d.text(x[i], y[i], z[i], str(txt + 1), color='black', fontsize=8)
        '''

        xyzt = coordinates_3d_with_time[class_member_mask & core_samples_mask]
        axis_3d.scatter(xyzt[:, 0]
                        , xyzt[:, 1]
                        , xyzt[:, 2]
                        , facecolor=tuple(col)
                        , s=250
                        , alpha=0.25)

        xyzt = coordinates_3d_with_time[class_member_mask & ~core_samples_mask]
        axis_3d.scatter(xyzt[:, 0]
                        , xyzt[:, 1]
                        , xyzt[:, 2]
                        , "o"
                        , facecolor=tuple(col)
                        , s=60
                        , alpha=0.25)
        if save_clusters:
            xyzt_and_diameter = coordinates_3d[class_member_mask]
            if k == -1:
                np.save(f'{TRACK_3D_NPY_BASE_FILENAME}_cluster_{k}_noise_points', xyzt_and_diameter)
            else:
                np.save(f'{TRACK_3D_NPY_BASE_FILENAME}_cluster_{k}', xyzt_and_diameter)


def log_n_clusters_and_n_noise(n_clusters_, n_noise_):
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script performs DBSCAN on the provided 3D track and plots the results.')

    parser.add_argument('--track-3d', '-ts3d',
                        type=str,
                        help='path to the 3D track segment npy-file',
                        default='./some_3d_track.npy')

    parser.add_argument('--dbscan-eps', '-eps',
                        type=float,
                        help='DBSCAN EPS parameter',
                        default=0.15)

    parser.add_argument('--dbscan-min-samples', '-ms',
                        type=int,
                        help='DBSCAN min_samples parameter',
                        default=10)

    args = parser.parse_args()

    # -- Change parameters here to avoid running the script from CLI if needed
    TRACK_3D_NPY_FILE_PATH = args.track_3d

    DBSCAN_EPS = args.dbscan_eps
    DBSCAN_MIN_SAMPLES = args.dbscan_min_samples

    # --------------------------------------------------------
    TRACK_3D_NPY_BASE_FILENAME = TRACK_3D_NPY_FILE_PATH[:-4]
    PLOT_NOISE_POINTS = False
    SAVE_CLUSTERS_TO_DISK = False

    main()
