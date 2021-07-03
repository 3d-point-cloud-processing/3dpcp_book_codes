import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
print("Loading a point cloud from", filename)
pcd = o3d.io.read_point_cloud(filename)
print(pcd)

labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / max(max_label,1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([pcd], zoom=0.8,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
