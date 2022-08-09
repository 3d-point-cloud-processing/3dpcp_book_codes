import numpy as np
import copy

import open3d as o3d

pcd = o3d.io.read_point_cloud("../data/tabletop_scene.ply")
o3d.visualization.draw_geometries([pcd], 
                                  zoom=0.459,
			                      front=[ 0.349, 0.063, -0.934 ],
			                      lookat=[ 0.039, 0.007, 0.524 ],
			                      up=[ -0.316, -0.930, -0.181 ]
                                  )

plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
                                         ransac_n=3,
                                         num_iterations=500)

[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

plane_cloud = pcd.select_by_index(inliers)
plane_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)

o3d.visualization.draw_geometries([plane_cloud, outlier_cloud],
                                  zoom=0.459,
			                      front=[ 0.349, 0.063, -0.934 ],
			                      lookat=[ 0.039, 0.007, 0.524 ],
			                      up=[ -0.316, -0.930, -0.181 ]
                                  )