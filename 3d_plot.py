import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
#centers_max_z1 = np.load('./Filtered8N.npy_cluster_1.npy')
#centers_max_z2 = np.load('./Filtered8N.npy_cluster_3.npy')

centers_max_z = np.load('./') #set here filepath
#centers_max_z = np.concatenate((centers_max_z1, centers_max_z2), axis=0)

#time_points = np.array([tp[-9:-4] for tp in centers_max_z[:, 3]], dtype=np.uint16)
time_points = np.array(centers_max_z[:, 3], dtype=np.float64)
time_pointsc = time_points.astype(int)
colors = cm.rainbow(np.linspace(0, 1, max(time_pointsc+1)))



centers_max_z_ = np.array(centers_max_z[:, 0:3], dtype=np.float64)
centers_max_z_[:,0:2] = centers_max_z_[:,0:2]/2#*(365/2048)
#centers_max_z_[:,0:2] = centers_max_z_[:,0:2]*(365/2048)
centers_max_z_[:,2] = centers_max_z_[:,2]*4
zero_array =  np.where(centers_max_z_ == 0)[0]
zero_array =  np.where(centers_max_z_ == 0)[0]
ninetyone_array =  np.where((centers_max_z_[:,2] < -881) | (centers_max_z_[:,2] > 600))[0]

centers_max_z_N = np.delete(centers_max_z_, zero_array, 0)
centers_max_z_N = np.delete(centers_max_z_, ninetyone_array, 0)
centers_max_z_NNN = centers_max_z_N
# Dimensions: 330 micrometer*330 micrometer + 400 micrometer





time_points = np.array(centers_max_z[:, 3], dtype=np.float64)
time_pointsc = time_points.astype(int)
time_pointsD = np.delete(time_pointsc, zero_array, 0)
time_pointsD = np.delete(time_pointsc, ninetyone_array, 0)

colored_time_points = [colors[tp] for tp in time_pointsD]


centers_with_time = np.column_stack([centers_max_z_, time_points])
centers_with_time_zero_filtered = np.column_stack([centers_max_z_NNN, time_pointsD])
#np.save('SecondDataSet_centers_new_better_Filter400_TRY_zero_filtered', centers_with_time_zero_filtered)

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')
#ax.set_xlim(0, 360)
ax.set_xlim(0, 1024)
#ax.set_ylim(0, 360)
ax.set_ylim(0, 1024)
#ax.set_zlim(-500, 1000)
ax.set_zlim(-00, 400)
#ax.scatter(centers_max_z_[:, 0], centers_max_z_[:, 1], centers_max_z_[:, 2], color=colored_time_points)# s=2, color=colored_time_points)


ax.set_xlabel("x [px]")
ax.set_ylabel("y [px]")
ax.set_zlabel("")
ax.set_zticklabels([])
#ax.set_zlabel("z [µm]")
#ax.plot(centers_max_z_[:, 0], centers_max_z_[:, 1], centers_max_z_[:, 2], color='r')
ax.scatter(centers_max_z_N[:, 0], centers_max_z_N[:, 1], centers_max_z_N[:, 2], s = 4, color=colored_time_points)
#ax.plot(centers_max_z_N[:, 0], centers_max_z_N[:, 1], centers_max_z_N[:, 2],color='r')
#cb = plt.colorbar(scat_plot + scat_plot2 , pad=.1)

#cb.set_ticklabels(["Time:25", "50", "75", "100", "125", "150"])
#ax.plot(centers_max_z_[:, 0], centers_max_z_[:, 1], centers_max_z_[:, 2], color='b')
# Plot and label original data
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(centers_max_z_[:, 0], centers_max_z_[:, 1], centers_max_z_[:, 2], s=5, color=colored_time_points, label='First Plot')

# Randomly re-order the data


# Plot and label re-ordered data
#ax.scatter(centers_max_z_2[:, 0], centers_max_z_2[:, 1], centers_max_z_2[:, 2], s=5, color=colored_time_points, label='Second Plot')
#ax.legend(loc='upper left')
ax.view_init(elev=270, azim=270)
#ax.view_init(elev=270-65, azim=270-30)
#ax.view_init(elev=40, azim=-166)
plt.show()


