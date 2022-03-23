import numpy as np
import copy

import open3d as o3d

def ComputeSphereCoefficient( p0, p1, p2, p3 ):
    """ 与えられた4点を通る球の方程式のパラメータ(a,b,c,r)を出力する．
        解が求まらない場合は，
    Args:
      p0,p1,p2,p3(numpy.ndarray): 4 points (x,y,z)
    Return:
      Sphere coefficients.
    """

    A = np.array([p0-p3,p1-p3,p2-p3])
    
    p3_2 = np.dot(p3,p3)
    b = np.array([(np.dot(p0,p0)-p3_2)/2,
                  (np.dot(p1,p1)-p3_2)/2,
                  (np.dot(p2,p2)-p3_2)/2])
    coeff = np.zeros(3)
    try:
        ans = np.linalg.solve(A,b)
    except:
        print( "!!Error!! Matrix rank is", np.linalg.matrix_rank(A) )
        print( "  Return", coeff )
        pass
    else:
        tmp = p0-ans
        r = np.sqrt( np.dot(tmp,tmp) )
        coeff = np.append(ans,r)

    return coeff

def EvaluateSphereCoefficient( pcd, coeff, distance_th=0.01 ):
    """ 球の方程式の係数の当てはまりの良さを評価する．
    Args:
      pcd(numpy.ndarray): Nx3 points
      coeff(numpy.ndarray): shpere coefficient. (a,b,c,r)
      distance_th(float):
    Returns:
      fitness: score [0-1]. larger is better
      inlier_dist: smaller is better
      inliers: indices of inliers
    """
    fitness = 0 # インライア点数/全点数
    inlier_dist = 0 #インライアの平均距離
    inliers = None #インライア点の番号セット
    
    dist = np.abs( np.linalg.norm( pcd - coeff[:3], axis=1 ) - coeff[3] )
    n_inlier = np.sum(dist<distance_th)
    if n_inlier != 0:
        fitness = n_inlier / pcd.shape[0]
        inlier_dist = np.sum((dist<distance_th)*dist)/n_inlier
        inliers = np.where(dist<distance_th)[0]
    
    return fitness, inlier_dist, inliers


if __name__ == "__main__":
    # データ読み込み
    pcd = o3d.io.read_point_cloud("../data/tabletop_scene1_segment.ply")
    np_pcd = np.asarray(pcd.points)
    o3d.visualization.draw_geometries([pcd])

    # Parameters
    ransac_n = 4 # 点群から選択する点数．球の場合は4．
    num_iterations = 1000 # RANSACの試行回数
    distance_th = 0.005 # モデルと点群の距離のしきい値
    max_radius = 0.05 # 検出する球の半径の最大値

    # 解の初期化
    best_fitness = 0 # モデルの当てはめの良さ．インライア点数/全点数
    best_inlier_dist = 10000.0 #インライア点の平均距離
    best_inliers = None # 元の点群におけるインライアのインデクス
    best_coeff = None # モデルパラメータ

    for n in range(num_iterations):
        c_id = np.random.choice( np_pcd.shape[0], 4, replace=False )
        coeff = ComputeSphereCoefficient( np_pcd[c_id[0]], np_pcd[c_id[1]], np_pcd[c_id[2]], np_pcd[c_id[3]] )
        if max_radius < coeff[3]:
            continue
        fitness, inlier_dist, inliers = EvaluateSphereCoefficient( np_pcd, coeff, distance_th )
        if (best_fitness < fitness) or ((best_fitness == fitness) and (inlier_dist<best_inlier_dist)):
            best_fitness = fitness
            best_inlier_dist = inlier_dist
            best_inliers = inliers
            best_coeff = coeff
            print(f"Update: Fitness = {best_fitness:.4f}, Inlier_dist = {best_inlier_dist:.4f}")

    if best_coeff is not None:
        print(f"Sphere equation: (x-{best_coeff[0]:.2f})^2 + (y-{best_coeff[1]:.2f})^2 + (z-{best_coeff[2]:.2f})^2 = {best_coeff[3]:.2f}^2")
    else:
        print("No sphere detected.")

    # 結果の可視化
    sphere_cloud = pcd.select_by_index(best_inliers)
    sphere_cloud.paint_uniform_color([0, 0, 1.0])
    outlier_cloud = pcd.select_by_index(best_inliers, invert=True)

    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=best_coeff[3])
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.3, 0.3, 0.7])
    mesh_sphere.translate(best_coeff[:3])
    o3d.visualization.draw_geometries([mesh_sphere]+[sphere_cloud+outlier_cloud])