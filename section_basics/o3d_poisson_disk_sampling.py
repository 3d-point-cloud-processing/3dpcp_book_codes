import sys
import numpy as np
import open3d as o3d

filename = sys.argv[1]
k = int(sys.argv[2])
print("Loading a triangle mesh from", filename)
mesh = o3d.io.read_triangle_mesh(filename)
print(mesh)

o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

downpcd = mesh.sample_points_poisson_disk(number_of_points=k)
print(downpcd)

o3d.visualization.draw_geometries([downpcd])
