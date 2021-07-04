import sys
import open3d as o3d
import numpy as np

filename = sys.argv[1]
print("Loading a point cloud from", filename)
mesh = o3d.io.read_triangle_mesh(filename)
print(mesh)
mesh.compute_vertex_normals()

# fit to unit cube
mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
o3d.visualization.draw_geometries([mesh])

print('mesh to voxel')
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.05)
o3d.visualization.draw_geometries([voxel_grid])

print('point cloud to voxel')
pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices
pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(np.array(pcd.points).shape[0], 3)))
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
o3d.visualization.draw_geometries([voxel_grid])

# check if points are within an occupied voxel
queries = np.asarray(pcd.points)
output = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
print(output[:10])
