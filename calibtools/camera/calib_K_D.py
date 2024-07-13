'''
Author: fanjin jinfan.@novauto.com.cn
Date: 2024-07-13 22:35:36
LastEditors: fanjin jinfan.@novauto.com.cn
LastEditTime: 2024-07-14 00:16:16
FilePath: /calibtools/calibtools/camera/calib_K_D.py
Description:  标定相机内参和畸变系数

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import os
import cv2
import glob
import argparse
import numpy as np
from typing import List
from calibtools.log import log_utils
from calibtools.camera.common import CamModel


logger = log_utils.get_logger()


def calib_fish_camera_charuco():
    parser = argparse.ArgumentParser(description='calib fish camera K&D use charuco', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--img_dir', type=str, help='image dir')
    parser.add_argument('-c', '--charuco_yaml_file', type=str, help='charuco yaml file')
    parser.add_argument('-o','--output_dir', type=str, help='output dir')
    parser.add_argument('-v', '--visilize', action='store_true', help='visualize calib')
    parser.add_argument('-s', '--save_intermediate_result', action='store_true', help='save intermediate result')
    
    params = parser.parse_args()
    img_dir = params.img_dir
    charuco_yaml_file = params.charuco_yaml_file
    output_dir = params.output_dir
    enable_visilize = params.visilize
    enable_save_intermediate_result = params.save_intermediate_result
    
    if not os.path.exists(charuco_yaml_file):
        logger.error('charuco_yaml_file not exists')
        return
    
    if not os.path.exists(img_dir):
        logger.error('img_dir not exists')
        return
    
    
    logger.info('calib_fish_camera_charuco start...')
    
    pattern = '/**/*.{jpg,png}'
    image_files = glob.glob(img_dir + '/**/*.{jpg,png}', recursive=True)
    
    logger.info('detect corners...')
    
    
    for path in image_files:
        logger.info(path)
        pass
    
    
    pass

    
def main():
    img_dir = "123"
    output_dir = "/home/frank/data/github/calibtools/data/calib_results"
    
    
    
    pass
    
    
if __name__ == '__main__':
    main()
    pass
    