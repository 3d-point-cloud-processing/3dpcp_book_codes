import open3d as o3d
import numpy as np
import sys

phi = (1+np.sqrt(5))/2
vertices = np.asarray([
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1],    
    [0, 1/phi, phi],
    [0, 1/phi, -phi],
    [0, -1/phi, phi],
    [0, -1/phi, -phi],
    [phi, 0, 1/phi],
    [phi, 0, -1/phi],
    [-phi, 0, 1/phi],
    [-phi, 0, -1/phi],    
    [1/phi, phi, 0],
    [-1/phi, phi, 0],
    [1/phi, -phi, 0],
    [-1/phi, -phi, 0]
])

i = 0
ROTATION_RADIAN_PER_PIXEL = 0.003
def rotate_view(vis):
    global i
    if i >= vertices.shape[0]:
        exit()
    vis.reset_view_point(True)
    ctr = vis.get_view_control()
    az = np.arctan2(vertices[i,1], vertices[i,0]) 
    el = np.arctan2(vertices[i,2], np.sqrt(vertices[i,0] * vertices[i,0] + vertices[i,1] * vertices[i,1]))
    ctr.rotate(-az/ROTATION_RADIAN_PER_PIXEL, el/ROTATION_RADIAN_PER_PIXEL)
    i += 1
    return False

pcd = o3d.io.read_point_cloud(sys.argv[1])

key_to_callback = {}
key_to_callback[ord(" ")] = rotate_view
o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
