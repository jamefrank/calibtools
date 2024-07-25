'''
Author: fanjin 
Date: 2024-07-25 10:49:40
LastEditors: fanjin 
LastEditTime: 2024-07-25 11:02:29
FilePath: /calibtools/calibtools/file_utils/np_to_cv_yaml.py
Description: 
numpy数组转为yaml文件,并存储到本地

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import cv2
import numpy as np


def save_corners(corners: np.ndarray, save_path: str):
    """
    将numpy数组转为yaml文件,并存储到本地
    """

    fs = cv2.FileStorage(save_path, cv2.FileStorage_WRITE)
    fs.write("corners", corners)
    fs.release()

    pass
