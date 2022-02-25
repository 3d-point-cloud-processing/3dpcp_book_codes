import sys
import numpy as np
import numpy.linalg as LA
import open3d as o3d

from icp_registration import ICPRegistration_PointToPoint
from icp_registration import ICPRegistration_PointToPlane
from icp_registration import visualize_icp_progress

mode = int(sys.argv[1])
pcd1 = o3d.io.read_point_cloud( "../data/bun000.pcd" )
pcd2 = o3d.io.read_point_cloud( "../data/bun045.pcd" )

pcd_s = pcd1.voxel_down_sample(voxel_size=0.003)
pcd_t = pcd2.voxel_down_sample(voxel_size=0.003)

# 初期状態の表示
pcd_s.paint_uniform_color([0.0, 1.0, 0.0])
pcd_t.paint_uniform_color([0.0, 0.0, 1.0])
o3d.visualization.draw_geometries([pcd_t, pcd_s] )

# ICPの実行．
reg = ICPRegistration_PointToPoint(pcd_s, pcd_t)
if mode == 1:
	print("Run Point-to-plane ICP algorithm")
	reg = ICPRegistration_PointToPlane(pcd_s, pcd_t)
reg.set_th_distance( 0.003 )
reg.set_n_iterations( 100 )
reg.set_th_ratio( 0.999 )
pcd_reg = reg.registration()

print("# of iterations: ", len(reg.d))
print("Registration ettor [m/pts.]:", reg.d[-1] )
print("Final transformation \n", reg.final_trans )

pcd_reg.paint_uniform_color([1.0,0.0,0.0])
o3d.visualization.draw_geometries([pcd_t, pcd_reg] )

# ICPの実行の様子の可視化
visualize_icp_progress( reg )