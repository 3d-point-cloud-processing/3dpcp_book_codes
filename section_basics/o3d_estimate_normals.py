import sys
import open3d as o3d
import numpy as np

filename = sys.argv[1]
print("Loading a point cloud from", filename)
mesh = o3d.io.read_triangle_mesh(filename)
print(mesh)
o3d.visualization.draw_geometries([mesh])

pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=10))

print(np.asarray(pcd.normals))
o3d.visualization.draw_geometries([pcd], point_show_normal=True)

mesh.compute_vertex_normals()

print(np.asarray(mesh.triangle_normals))
o3d.visualization.draw_geometries([mesh])
