'''
Author: fanjin 
Date: 2024-07-19 15:33:04
LastEditors: fanjin 
LastEditTime: 2024-07-25 16:07:03
FilePath: /calibtools/calibtools/img_utils/lidar_project_img.py
Description: 

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import os
import cv2
import math
import numpy as np
from pyntcloud import PyntCloud
from pc_common.point_cloud.point_cloud import PointCloud


def project_pcd_to_img(img_path: str, pcd_path: str, lidar2camera, K, output_dir: str):
    img = cv2.imread(img_path)
    # cloud = PyntCloud.from_file(pcd_path)
    # xyz = cloud.xyz

    cloud = PointCloud.read_file(pcd_path)
    xyz = cloud.points
    # xyz = xyz[xyz[:, 0] <= 5]
    # xyz = xyz[xyz[:, 0] >= 20]
    # xyz = xyz[xyz[:, 0] <= 55]

    rvec, tvec = cv2.Rodrigues(lidar2camera[:3, :3])[0], lidar2camera[:3, 3]

    # 生成颜色映射表
    color_cnt = 2048
    gray = np.arange(0, color_cnt, 1, dtype=np.uint8)

    color_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Create a list of RGB values from the color map
    color_list = []
    for i in range(color_map.shape[0]):
        for j in range(color_map.shape[1]):
            bgr_value = color_map[i, j]
            rgb_value = (int(bgr_value[2]), int(bgr_value[1]), int(bgr_value[0]))
            color_list.append(rgb_value)

    # 查找xyz中深度最大最小值
    depth_min = np.Inf
    depth_max = -1*np.Inf
    for item in xyz:
        depth = math.sqrt(item[0]*item[0]+item[1]*item[1])
        if depth < depth_min:
            depth_min = depth
        if depth > depth_max:
            depth_max = depth

    imgpts1, _ = cv2.projectPoints(xyz, rvec, tvec, K, None)
    for i, item in enumerate(imgpts1):
        x = item[0][0]
        y = item[0][1]
        x1 = xyz[i][0]
        y1 = xyz[i][1]

        # if x >= 0 and x < 1920 and y >= 0 and y < 1080:  #TODO
        if x >= 0 and x < 3840 and y >= 0 and y < 2160:
            depth = math.sqrt(x1*x1+y1*y1)
            depth = int((depth-depth_min)/(depth_max-depth_min)*(color_cnt-1))
            cv2.circle(img, (int(x), int(y)), 3, color_list[depth], -1)

    cv2.imwrite(os.path.join(output_dir, "test.jpg"), img)
