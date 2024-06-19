'''
Author: fanjin jinfan.@novauto.com.cn
Date: 2024-06-18 22:33:44
LastEditors: fanjin 
LastEditTime: 2024-06-19 10:15:29
FilePath: /calibtools/python/lidar2gnss.py
Description: 利用opencv的手眼标定实现  lidar和gnss外参的标定

# conda 环境创建
conda create -n handeye python=3.8
pip install pyquaternion
pip install opencv-python==4.5.1.48 -i https://pypi.tuna.tsinghua.edu.cn/simple --verbose
pip install opencv-contrib-python==4.5.1.48 -i https://pypi.tuna.tsinghua.edu.cn/simple --verbose

# txt 内容格式：
frame_id timestamp x y z qx qy qz qw

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import cv2
import numpy as np
from typing import List
from pyquaternion import Quaternion

def load_pose_txt(pose_txt:str):
    tss = []
    quats = []
    trans = []

    with open(pose_txt, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            data = line.split()
            # print(data)
            data = [float(x) for x in data]
            
            quat = Quaternion(data[8],data[5],data[6],data[7])
            tran = np.array([data[2],data[3],data[4]],dtype=float)
            
            tss.append(data[1])
            quats.append(quat)
            trans.append(tran)
            
            # rot_mat = quat.rotation_matrix
            # pose = np.eye(4)
            # pose[:3,:3] = rot_mat
            # pose[3,:3] = np.array([data[2],data[3],data[4]],dtype=float)
            # print(pose)
            # poses.append(pose)
    
    return tss,quats,trans

def align_two_poses(base_tss, other_tss, other_quats, other_trans):
    assert base_tss[0] > other_tss[0]
    assert base_tss[-1] < other_tss[-1]
    
    other_align_quats = []
    other_align_trans = []
    idx = 0
    for ts in base_tss:
        while idx<len(other_tss)-1 and (not(ts>=other_tss[idx] and ts<=other_tss[idx+1])):
            idx += 1
        alpha = (ts-other_tss[idx])/(other_tss[idx+1]-other_tss[idx])
        q_interp = Quaternion.slerp(other_quats[idx], other_quats[idx+1], alpha)
        t_interp = other_trans[idx] + (other_trans[idx+1]-other_trans[idx])*alpha
        other_align_quats.append(q_interp)
        other_align_trans.append(t_interp)
        pass
    
    return base_tss, other_align_quats, other_align_trans

def poses_to_mats(quats, trans):
    mats = []
    for quat,tran in zip(quats, trans):
        rot_mat = quat.rotation_matrix
        pose = np.eye(4)
        pose[:3,:3] = rot_mat
        pose[:3,3] = tran
        mats.append(pose)
    return mats

def mats_to_relative_mats(mats: List[np.ndarray]):
    first_mat = mats[0]
    first_mat_inv = np.linalg.inv(first_mat)
    
    relative_mats = []
    for mat in mats:
        r_mat = first_mat_inv@mat
        relative_mats.append(r_mat)
        
    return relative_mats

def mats_to_Rs_ts(mats: List[np.ndarray]):
    Rs = []
    ts = []
    for mat in mats:
        R = mat[:3,:3]
        t = mat[:3,3]
        Rs.append(R)
        ts.append(t)
        
    return Rs,ts

def calib_lidar_2_gnss(lidar_pose_txt:str, gnss_pose_txt:str):
    l_tss, l_quats, l_trans = load_pose_txt(lidar_pose_txt)
    g_tss, g_quats, g_trans = load_pose_txt(gnss_pose_txt)
    
    g_align_tss, g_align_quats, g_align_trans = align_two_poses(l_tss, g_tss, g_quats, g_trans)
    l_mats = poses_to_mats(l_quats, l_trans)
    g_mats = poses_to_mats(g_align_quats, g_align_trans)
    
    l_r_mats = mats_to_relative_mats(l_mats)
    g_r_mats = mats_to_relative_mats(g_mats)
    
    l_r_mats_inv = [np.linalg.inv(mat) for mat in l_r_mats]
    
    l_Rs, l_ts = mats_to_Rs_ts(l_r_mats_inv)
    g_Rs, g_ts = mats_to_Rs_ts(g_r_mats)
    
    # methodHE:  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gad10a5ef12ee3499a0774c7904a801b99
    # methodHE = [cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_ANDREFF, cv2.CALIB_HAND_EYE_DANIILIDIS]
    MTH = cv2.CALIB_HAND_EYE_TSAI
    
    # for MTH in methodHE:
    R_lidar2gnss, t_lidar2gnss = cv2.calibrateHandEye(g_Rs, g_ts, l_Rs, l_ts, None, None, MTH)
    
    print("lidar2gnss:")
    print(R_lidar2gnss)
    print(t_lidar2gnss)

def main():
    lidar_pose_txt = "/home/frank/data/GitLab/g3_calib/data/imu_top_jx_lidar/2024-06-17-14-31-38/lidar-poses.txt"
    gnss_pose_txt = "/home/frank/data/GitLab/g3_calib/data/imu_top_jx_lidar/2024-06-17-14-31-38/2024-06-17-14-31-38.txt"
    # load_pose_txt(lidar_pose_txt)
    
    #
    calib_lidar_2_gnss(lidar_pose_txt, gnss_pose_txt)
    
    pass
 
 
if __name__ == '__main__':
    main()
    pass
