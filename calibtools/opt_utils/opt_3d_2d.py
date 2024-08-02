'''
Author: fanjin 
Date: 2024-07-19 14:59:44
LastEditors: fanjin 
LastEditTime: 2024-08-02 11:33:26
FilePath: /calibtools/calibtools/opt_utils/opt_3d_2d.py
Description: solvepnp

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import cv2
import numpy as np


def calib_3d_to_2d(car_pts, img_pts, K):
    # _, rvec, tvec, _ = cv2.solvePnPRansac(car_pts, img_pts, K, np.array([0,0,0,0],dtype=float))
    _, rvec, tvec = cv2.solvePnP(car_pts, img_pts, K, np.array([0, 0, 0, 0], dtype=np.float64))

    rot_mat, _ = cv2.Rodrigues(rvec)
    transformation_matrix = np.hstack((rot_mat, tvec))
    transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))
    print("变换矩阵:")
    print(transformation_matrix)

    reprojected_points, _ = cv2.projectPoints(
        car_pts, rvec, tvec, K, np.array([0, 0, 0, 0], dtype=np.float64)
    )

    # 计算重投影误差
    reprojection_error = np.mean(
        np.linalg.norm(img_pts - reprojected_points.squeeze(), axis=1)
    )
    print(np.linalg.norm(img_pts - reprojected_points.squeeze(), axis=1))
    print("重投影误差:")
    print(reprojection_error)

    return transformation_matrix
