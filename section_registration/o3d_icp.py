import open3d as o3d
import numpy as np
import copy


pcd1 = o3d.io.read_point_cloud( "../data/bun000.pcd" )
pcd2 = o3d.io.read_point_cloud( "../data/bun045.pcd" )

pcd_s = pcd1.voxel_down_sample(voxel_size=0.003)
pcd_t = pcd2.voxel_down_sample(voxel_size=0.003)

# 初期状態の表示
pcd_s.paint_uniform_color([0.0, 1.0, 0.0])
pcd_t.paint_uniform_color([0.0, 0.0, 1.0])
o3d.visualization.draw_geometries([pcd_s,pcd_t])

# ICPによる位置合わせ
threshold = 0.05
trans_init = np.identity(4)
obj_func = o3d.pipelines.registration.TransformationEstimationPointToPoint()
result = o3d.pipelines.registration.registration_icp( pcd_s, pcd_t,
                                                      threshold,
                                                      trans_init,
                                                      obj_func
                                                    )

trans_reg = result.transformation
print(trans_reg) 

# 得られた変換行列を点群に適用
pcd_reg = copy.deepcopy(pcd_s).transform(trans_reg)
pcd_reg.paint_uniform_color([1.0, 0.0, 0.0])
o3d.visualization.draw_geometries([pcd_reg,pcd_t])