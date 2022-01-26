import open3d as o3d
import copy
import numpy as np
import numpy.linalg as LA

class ICPRegistration:
    # input source, target
    def __init__( self, pcd_s, pcd_t ):
        self.pcd_s = copy.deepcopy(pcd_s)
        self.pcd_t = copy.deepcopy(pcd_t)
        self.n_points = len(self.pcd_s.points)
        
        self.pcds = []
        self.d = []
        self.final_trans = np.identity(4)
        self.n_iteration = 30
        self.th_distance = 0.001
        self.th_ratio = 0.999
        
        # 初期化ステップ
        ## レジストレーションベクトル
        self.q = np.array([1.,0.,0.,0.,0.,0.,0.])
        
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd_t)
        self.np_pcd_t = np.asarray(self.pcd_t.points)

    def set_n_iterations( self, n ):
        self.n_iteration = n

    def set_th_distance( self, th ):
        self.th_distance = th

    def set_th_ratio( self, th ):
        self.th_ratio = th
        
    def closest_points( self ):
        idx_list = []
        distance = []
        for i in range(self.n_points):
            [k, idx, d] = self.pcd_tree.search_knn_vector_3d(self.pcd_s.points[i], 1)
            idx_list.append(idx[0])
            distance.append(d[0])

        np_pcd_y = self.np_pcd_t[idx_list]
        self.d.append( np.sqrt(np.mean(np.array(distance))) )
        return np_pcd_y.copy()
    
    def compute_registration_param( self, np_pcd_y ):
        # 重心の取得
        mu_s = self.pcd_s.get_center()
        mu_y = np.mean(np_pcd_y, axis=0)

        # 共分散行列の計算
        np_pcd_s = np.asarray(self.pcd_s.points)
        covar = np.zeros( (3,3) )
        n_points = np_pcd_s.shape[0]
        for i in range(n_points):
            covar += np.dot( np_pcd_s[i].reshape(-1, 1), np_pcd_y[i].reshape(1, -1) )
        covar /= n_points
        covar -= np.dot( mu_s.reshape(-1,1), mu_y.reshape(1,-1) )
        
        ## anti-symmetrix matrix
        A = covar - covar.T
        delta = np.array([A[1,2],A[2,0],A[0,1]])
        tr_covar = np.trace(covar)
        i3d = np.identity(3)
        
        # symmetric matrixの作成
        Q = np.zeros((4,4))
        Q[0,0] = tr_covar
        Q[0,1:4] = delta
        Q[1:4,0] = delta
        Q[1:4,1:4] = covar + covar.T - tr_covar*i3d
        
        w, v = LA.eig(Q)
        rot = self.quaternion2rotation(v[:,np.argmax(w)])
        trans = mu_y - np.dot(rot,mu_s)
        
        self.q = np.concatenate((v[np.argmax(w)],trans))
        transform = np.identity(4)
        transform[0:3,0:3] = rot.copy()
        transform[0:3,3] = trans.copy()
        return transform
    
        
    def registration( self ):
        
        for i in range(self.n_iteration):
            # Step. 1
            np_pcd_y = self.closest_points()
            # Step. 2
            transform = self.compute_registration_param( np_pcd_y )
            # Step. 3
            self.pcd_s.transform(transform)
            self.final_trans = np.dot(transform,self.final_trans)
            self.pcds.append(copy.deepcopy(self.pcd_s))
            # Step. 4
            if ((2<i) and (self.th_ratio < self.d[-1]/self.d[-2])) or (self.d[-1] < self.th_distance):
                break
        
        return self.pcd_s
        
    # 四元数を回転行列に変換する
    def quaternion2rotation( self, q ):
        rot = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 
                         2.0*(q[1]*q[2]-q[0]*q[3]), 
                         2.0*(q[1]*q[3]+q[0]*q[2])],

                        [2.0*(q[1]*q[2]+q[0]*q[3]),
                        q[0]**2+q[2]**2-q[1]**2-q[3]**2,
                         2.0*(q[2]*q[3]-q[0]*q[1])],

                        [2.0*(q[1]*q[3]-q[0]*q[2]),
                         2.0*(q[2]*q[3]+q[0]*q[1]),
                        q[0]**2+q[3]**2-q[1]**2-q[2]**2]]
                      )
        return rot