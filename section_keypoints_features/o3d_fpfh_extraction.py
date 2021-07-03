import sys
import open3d as o3d

filename = sys.argv[1]
print("Loading a point cloud from", filename)
pcd = o3d.io.read_point_cloud(filename)
print(pcd)

pcd.estimate_normals(
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))

fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=100))

print(fpfh)
print(fpfh.data)
