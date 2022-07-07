import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

dirname = sys.argv[ 1 ]

source_color = o3d.io.read_image(dirname + "/RGBD/color/00000.jpg")
source_depth = o3d.io.read_image(dirname + "/RGBD/depth/00000.png")
target_color = o3d.io.read_image(dirname + "/RGBD/color/00001.jpg")
target_depth = o3d.io.read_image(dirname + "/RGBD/depth/00001.png")
source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    source_color, source_depth)
target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    target_color, target_depth)

plt.subplot(2, 2, 1)
plt.title("Source grayscale image")
plt.imshow(source_rgbd.color)
plt.subplot(2, 2, 2)
plt.title("Source depth image")
plt.imshow(source_rgbd.depth)
plt.subplot(2, 2, 3)
plt.title("Target grayscale image")
plt.imshow(target_rgbd.color)
plt.subplot(2, 2, 4)
plt.title("Target depth image")
plt.imshow(target_rgbd.depth)
plt.show()

pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
    dirname + "/camera_primesense.json")
print(pinhole_camera_intrinsic.intrinsic_matrix)

option = o3d.pipelines.odometry.OdometryOption()
odo_init = np.identity(4)
print(option)

[success_term, trans_term,
 info] = o3d.pipelines.odometry.compute_rgbd_odometry(
     source_rgbd, target_rgbd, pinhole_camera_intrinsic, odo_init,
     o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
     #o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

if not success_term:
    exit()
source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    source_rgbd, pinhole_camera_intrinsic)
target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    target_rgbd, pinhole_camera_intrinsic)
print("Using RGB-D Odometry")
print(trans_term)
source_pcd.transform(trans_term)
o3d.visualization.draw_geometries([target_pcd, source_pcd],
                                  zoom=0.48,
                                  front=[0.0999, -0.1787, -0.9788],
                                  lookat=[0.0345, -0.0937, 1.8033],
                                  up=[-0.0067, -0.9838, 0.1790])
