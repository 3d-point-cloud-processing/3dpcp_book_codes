import open3d as o3d
import numpy as np
import copy

def dist( p, X ):
    dists = np.linalg.norm(p-X,axis=1) 
    min_dist = min(dists)
    min_idx = np.argmin(dists)
    
    return min_dist, min_idx

# point cloud as sin function
X_x = np.arange(-np.pi,np.pi, 0.1)
X_y = np.sin(X_x)
X_z = np.zeros(X_x.shape)
X = np.vstack([X_x, X_y, X_z]).T

# point p
p = np.array([1.0,0.0,0.0])

# open3d point cloud of X
pcd_X = o3d.geometry.PointCloud()
pcd_X.points = o3d.utility.Vector3dVector(X)
pcd_X.paint_uniform_color([0.5,0.5,0.5])

# open3d point cloud of p
pcd_p = o3d.geometry.PointCloud()
pcd_p.points = o3d.utility.Vector3dVector([p])
pcd_p.paint_uniform_color([0.0,0.0,1.0])

# Create axis
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

# Visualization
o3d.visualization.draw_geometries([mesh, pcd_X,pcd_p])


min_dist, min_idx = dist(p,X)
np.asarray(pcd_X.colors)[min_idx] = [0.0,1.0,0.0]
print("distance:{}, idx:{}".format(min_dist, min_idx))
o3d.visualization.draw_geometries([mesh,pcd_X,pcd_p])