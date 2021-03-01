import sys
import open3d as o3d
import numpy as np
from keypoints_to_spheres import keypoints_to_spheres

def compute_harris3d_keypoints( pcd, radius=0.01, max_nn=10, threshold=0.001 ):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    harris = np.zeros( len(np.asarray(pcd.points)) )
    is_active = np.zeros( len(np.asarray(pcd.points)), dtype=bool )
    for i in range( len(np.asarray(pcd.points)) ):
        [num_nn, inds, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], max_nn)
        pcd_normals = pcd.select_by_index(inds)
        pcd_normals.points = pcd_normals.normals
        [_, covar] = pcd_normals.compute_mean_and_covariance()
        harris[ i ] = np.linalg.det( covar ) / np.trace( covar )
        #print (harris[i])
        if (harris[ i ] > threshold):
            is_active[ i ] = True

    # NMS
    for i in range( len(np.asarray(pcd.points)) ):
        if is_active[ i ]:
            [num_nn, inds, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], max_nn)
            inds.pop( harris[inds].argmax() )
            is_active[ inds ] = False

    print (np.sum(is_active))
    #exit()
    keypoints = pcd.select_by_index(np.where(is_active)[0])
    return keypoints

# main
filename = sys.argv[1]
print("Loading a point cloud from", filename)
pcd = o3d.io.read_point_cloud(filename)
print(pcd)

keypoints = compute_harris3d_keypoints( pcd )

pcd.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([keypoints_to_spheres(keypoints), pcd])
