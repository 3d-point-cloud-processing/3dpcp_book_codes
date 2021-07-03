import sys
import open3d as o3d
import numpy as np

def extract_fpfh( filename ):
    print (" ", filename)
    pcd = o3d.io.read_point_cloud(filename)
    pcd = pcd.voxel_down_sample(0.01)
    pcd.estimate_normals(
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=10))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=100))
    sum_fpfh = np.sum(np.array(fpfh.data),1)
    return( sum_fpfh / np.linalg.norm(sum_fpfh) )

# main
nsamp = 100
dirname = "rgbd-dataset"
classes = ["apple", "banana", "camera"]
feat_train = np.zeros( (len(classes), nsamp, 33) )
feat_test = np.zeros( (len(classes), nsamp, 33) )

# Extract features
for i in range(len(classes)):
    print ("Extracting train features in " + classes[i] + "...")
    for n in range(nsamp):
        filename = dirname + "/" + classes[i] + "/" + classes[i] + \
                   "_1/" + classes[i] + "_1_1_" + str(n+1) + ".pcd"
        feat_train[ i, n ] = extract_fpfh( filename )
    print ("Extracting test features in " + classes[i] + "...")
    for n in range(nsamp):
        filename = dirname + "/" + classes[i] + "/" + classes[i] + \
                   "_1/" + classes[i] + "_1_4_" + str(n+1) + ".pcd"
        feat_test[ i, n ] = extract_fpfh( filename )

# 1-NN classification
for i in range(len(classes)):
    max_sim = np.zeros((3, nsamp))
    for j in range(len(classes)):
        sim = np.dot(feat_test[i], feat_train[j].transpose())
        max_sim[j] = np.max(sim,1)
    correct_num = (np.argmax(max_sim,0) == i).sum()
    print ("Accuracy of", classes[i], ":", correct_num*100/nsamp, "%")
