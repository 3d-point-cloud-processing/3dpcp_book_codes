import sys
import open3d as o3d
from keypoints_to_spheres import keypoints_to_spheres

filename = sys.argv[1]
print("Loading a point cloud from", filename)
pcd = o3d.io.read_point_cloud(filename)
print(pcd)

keypoints = \
o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                            salient_radius=0.005,
                                            non_max_radius=0.005,
                                            gamma_21=0.5,
                                            gamma_32=0.5)

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))
pcd.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([keypoints_to_spheres(keypoints), pcd])
