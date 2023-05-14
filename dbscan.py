import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib as mpl

MHI_TIME_TOLERANCE = 5


def weighted_point(p):
    wp = np.array(p[0:4])
    # scale z axis
    wp[2] = 4 * wp[2]

    return wp


def custom_distance(p1, p2):
    dist = np.linalg.norm(weighted_point(p1) - weighted_point(p2))

    return dist


def main():
    # see https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    data_filename = f'{MHI_TIME_TOLERANCE}' #set here filename
    centers_max_z = np.load(f'.//{data_filename}.npy').astype(float) #set here filepath

    # convert pixels to micro meter
    centers_max_z[:, 0:2] = centers_max_z[:, 0:2] * (330.0 / 2048)  # convert X-Y from pixel to micro meter values
    centers_max_z[:, 2] = centers_max_z[:, 2] * 4  # convert z to correct micro meter values

    time = centers_max_z[:, 3]
    centers_max_z_ = centers_max_z[:, 0:3]

    centers_with_time = centers_max_z[:, :4]
    centers_with_time_and_diameter = centers_max_z

    '''
    z = centers_max_z[:, 3]
    fig, ax = plt.subplots()

    ax.scatter(range(0, time[20:400].size), time[20:400])
    plt.show()
    return
    ax.scatter(time, z)

    ax.set_xlabel('time')
    ax.set_ylabel('z')
    ax.set_title('z-time plot')

    # show the plot
    plt.show()

    return
   '''

    '''
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(centers_with_time[:, 2:4])
    distances, _ = nn.kneighbors(centers_with_time[:, 2:4])


    kl = KneeLocator(np.arange(len(distances)), np.sort(distances[:, 1]), curve='convex', direction='increasing')
    max_eps = np.sort(distances[:, 1])[kl.knee]
    
    plt.plot(np.sort(distances[:, 1]))
    plt.show()
    return
    '''

    # calculate the norm of each column
    max_values = np.max(centers_with_time, axis=0)

    # normalize the columns
    normalized_centers_with_time = centers_with_time / max_values

    db = DBSCAN(eps=0.15, min_samples=10, metric=custom_distance) #set here parameters for DBSCAN
    db.fit(normalized_centers_with_time)

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    label_colors = [plt.cm.rainbow(each) for each in np.linspace(0, 1, len(unique_labels))]

    time_colors = plt.cm.rainbow(np.linspace(0, 1, int(max(time)) + 1))
    time_points_colors = [time_colors[int(tp)] for tp in time]

    # setup 3d plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 365)
    ax.set_ylim(0, 365)
    ax.set_zlim(0, 400)

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_zlabel("z [µm]")
    ax.view_init(elev=40, azim=-166)

    '''
    mask = labels == -1
    xyzt = centers_with_time[~mask & core_samples_mask]
    no_noise_time_points_colors = np.array(time_points_colors)[~mask & core_samples_mask]
    ax.scatter(xyzt[:, 0]
               , xyzt[:, 1]
               , xyzt[:, 2]
               , "o"
               , s=10
               , facecolor=no_noise_time_points_colors
               , edgecolors=no_noise_time_points_colors)
    '''
    sc = ax.scatter(centers_with_time[:, 0]
               , centers_with_time[:, 1]
               , centers_with_time[:, 2]
               , "o"
               , s=10
               #, facecolor=time_points_colors
               #, edgecolors=time_points_colors)
               , facecolor=time_points_colors
               , edgecolors=time_points_colors)

    custom_cmap = ListedColormap(time_points_colors)
    # Create a custom Normalize object to map your colors to a range of values
    norm = mpl.colors.Normalize(vmin=0, vmax=len(time_points_colors) - 1)

    # Create a ScalarMappable object with the custom colormap and normalization
    sm = ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])

    # Create a color bar using the ScalarMappable object
    cbar = plt.colorbar(sm, orientation='horizontal')
    cbar.set_label('Time index', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.show()
    return
    # plt.savefig(f"plots/no_dsscan_first_data_set_time_tolerance_{MHI_TIME_TOLERANCE}.png", format="png")

    for k, col in zip(unique_labels, label_colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            continue

        class_member_mask = labels == k

        xyzt = centers_with_time[class_member_mask & core_samples_mask]

        X = xyzt[:, 0]
        Y = xyzt[:, 1]
        Z = xyzt[:, 2]

        scatter = ax.scatter(
            X
            , Y
            , Z
            , facecolor=tuple(col)
            # , edgecolors="k"
            , s=250
            , alpha=0.25
        )
        for i, txt in enumerate(labels[class_member_mask & core_samples_mask]):
            ax.text(X[i], Y[i], Z[i], str(txt + 1), color='black', fontsize=8)

        xyzt = centers_with_time[class_member_mask & ~core_samples_mask]
        ax.scatter(
            xyzt[:, 0]
            , xyzt[:, 1]
            , xyzt[:, 2]
            , "o"
            , facecolor=tuple(col)
            # , edgecolors="k"
            , s=60
            , alpha=0.25
        )

        # xyzt_and_diameter = centers_with_time_and_diameter[class_member_mask]
        # if k == -1:
        #     np.save(f"./cluster_data/{data_filename}_cluster_{k}_noise", xyzt_and_diameter)
        # np.save(f"./cluster_data/{data_filename}_cluster_{k}", xyzt_and_diameter)

    # create a legend with label colors
    handles = [plt.Rectangle((0, 0), 1, 1, fc=tuple(label_colors[i])) for i in range(len(unique_labels))]
    labels = [f'Track {i + 1}' for i in range(len(unique_labels) - 1)]
    ax.legend(handles, labels, framealpha=0.0)

    # plt.title(f"Estimated number of clusters: {n_clusters_}\n"
    #          f"Estimated number of noise points: {n_noise_}")

    plt.savefig(f"plots/dsscan_first_data_set_MHI_time_tolerance_{MHI_TIME_TOLERANCE}_clusters_{n_clusters_}_noise_points_{n_noise_}.png", format="png")
    # plt.savefig(f"plots/dsscan_first_data_set_MHI_time_tolerance_{MHI_TIME_TOLERANCE}_clusters_{n_clusters_}_noise_points_{n_noise_}_no_noise_points.png", format="png")

    plt.show()


if __name__ == "__main__":
    main()
