import numpy as np
import open3d as o3d
import copy

# Create axes
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

# Rotate
R = o3d.geometry.get_rotation_matrix_from_yxz([np.pi/3, 0, 0])
print ("R:",np.round(R,7))
R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi/3, 0])
print ("R:",np.round(R,7))
R = o3d.geometry.get_rotation_matrix_from_quaternion([np.cos(np.pi/6), 0, np.sin(np.pi/6), 0])
print ("R:",np.round(R,7))
mesh_r = copy.deepcopy(mesh)
mesh_r.rotate(R, center=[0,0,0])

# Translate
t = [0.5, 0.7, 1]
mesh_t = copy.deepcopy(mesh_r).translate(t)
print ("Type q to continue.")
o3d.visualization.draw_geometries([mesh, mesh_t])

# Rotate and translate
T = np.eye(4)
T[:3,:3] = R
T[:3,3] = t
mesh_t = copy.deepcopy(mesh).transform(T)
print ("Type q to continue.")
o3d.visualization.draw_geometries([mesh, mesh_t])

# Scale
mesh_s = copy.deepcopy(mesh_t)
mesh_s.scale(0.5, center=mesh_s.get_center())
print ("Type q to exit.")
o3d.visualization.draw_geometries([mesh, mesh_s])
