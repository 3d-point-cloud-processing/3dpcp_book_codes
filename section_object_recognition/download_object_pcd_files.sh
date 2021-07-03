#!/bin/bash

url=https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii

for c in apple banana camera
do
    wget $url/${c}_1.tar
    tar xvf ${c}_1.tar
done
