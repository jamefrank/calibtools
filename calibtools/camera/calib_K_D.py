'''
Author: fanjin jinfan.@novauto.com.cn
Date: 2024-07-13 22:35:36
LastEditors: fanjin jinfan.@novauto.com.cn
LastEditTime: 2024-07-15 23:16:58
FilePath: /calibtools/calibtools/camera/calib_K_D.py
Description:  标定相机内参和畸变系数

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import os
import cv2
import glob
import yaml
import sys
import json
import argparse
import numpy as np
import cv2.aruco as aruco
from typing import List
from calibtools.log import log_utils
from calibtools.camera.common import CamModel


logger = log_utils.get_logger()


def calib_camera_charuco():
    parser = argparse.ArgumentParser(description='calib fish camera K&D use charuco', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--img_dir', type=str, help='image dir')
    parser.add_argument('-c', '--charuco_yaml_file', type=str, help='charuco yaml file')
    parser.add_argument('-o','--output_dir', type=str, help='output dir')
    parser.add_argument('-s', '--save_intermediate_result', action='store_true', help='save intermediate result')
    parser.add_argument('-m', '--model', choices=['fisheye', 'pinhole'], type=str, help='camera model')
    
    params = parser.parse_args()
    img_dir = params.img_dir
    charuco_yaml_file = params.charuco_yaml_file
    output_dir = params.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'temp'), exist_ok=True)
    
    enable_save_intermediate_result = params.save_intermediate_result
    cam_model = None
    if "fisheye" == params.model:
        cam_model = CamModel.FISH
    elif "pinhole" == params.model:
        cam_model = CamModel.PINHOLE
    
    if not os.path.exists(charuco_yaml_file):
        logger.error('charuco_yaml_file not exists')
        return
    
    if not os.path.exists(img_dir):
        logger.error('img_dir not exists')
        return
    
    #
    with open(charuco_yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    dict_type = None
    if 'DICT_4X4_250' == data['dict']:
        dict_type = aruco.DICT_4X4_250
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    squaresX = data['squaresX']
    squaresY = data['squaresY']
    squareLength = data['squareLength']/1000
    markerLength = data['markerLength']/1000
    board = aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, aruco_dict)

    # 
    logger.info('read img paths...')
    file_extensions = ["*.jpg", "*.png"]
    image_files = []
    for ext in file_extensions:
        image_files.extend(glob.glob(os.path.join(img_dir, ext)))
    
    
    #
    logger.info('detect corners...')
    all_corners = []
    all_ids = []
    all_paths = []
    image_size = None
    for path in image_files:
        logger.info(path)
        img = cv2.imread(path)
        if image_size is None:
            image_size = img.shape[:2]
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        # aruco.drawDetectedMarkers(img, corners, ids)
        charuco_corners = None
        charuco_ids = None
        if len(corners) > 0:
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, img, board)
            if charuco_corners is not None and len(charuco_corners) == (squaresX-1)*(squaresY-1):
                # aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
                all_corners.append(charuco_corners)    
                all_ids.append(charuco_ids)
                all_paths.append(path)
                    
    # 
    obj_points = []  # 存储三维物体点的列表
    for j in range(squaresY-1):
        for i in range(squaresX-1):
            point = []
            point.append((i+1)*squareLength)
            point.append((j+1)*squareLength)
            point.append(0)
            obj_points.append(point)
    obj_points = np.array(obj_points, dtype=np.float32)
    obj_points = np.expand_dims(obj_points, axis=1)
    all_obj_points = [obj_points]*len(all_corners)
    
    
    #
    if cam_model == CamModel.FISH:
        flags = cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT  | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9)
        retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(all_obj_points, all_corners, image_size, None, None, flags=flags, criteria=criteria)
    elif cam_model == CamModel.PINHOLE:
        flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 | cv2.CALIB_FIX_S1_S2_S3_S4  | cv2.CALIB_FIX_TAUX_TAUY
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9)
        retval, K, D, rvecs, tvecs = aruco.calibrateCameraCharuco(all_corners, all_ids, board, image_size, None, None, None, None, flags=flags, criteria=criteria)
    
    #
    calib_res = {}
    calib_res['K'] = K.tolist()
    calib_res['D'] = D.squeeze().tolist()
    calib_res['W'] = image_size[1]
    calib_res['H'] = image_size[0]
    calib_res['Error'] = retval
    calib_res['temp'] = []
    
    #
    for i,obj_points in enumerate(all_obj_points):
        img = cv2.imread(all_paths[i])
        aruco.drawDetectedCornersCharuco(img, all_corners[i], all_ids[i])
        if cam_model == CamModel.FISH:
            img_points = cv2.fisheye.projectPoints(obj_points, rvecs[i], tvecs[i], K, D)[0]
        else:
            img_points = cv2.projectPoints(obj_points, rvecs[i], tvecs[i], K, D)[0]
        img_points = img_points.reshape(-1, 2)
        corners = all_corners[i].reshape(-1, 2)
        error = cv2.norm(img_points, corners, cv2.NORM_L2) / len(img_points)
        logger.info(f'{i}: {all_paths[i]}, error: {error}')
        
        cv2.putText(img, f'Error: {error:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for pt in img_points:
            cv2.circle(img, (int(pt[0]),int(pt[1])), 5, (0, 0, 255), 1)
        if enable_save_intermediate_result:
            cv2.imwrite(f'{output_dir}/temp/{os.path.basename(all_paths[i])}.jpg', img)
            
        calib_res['temp'].append(
            {
                'error': error,
                'img_path': all_paths[i],
                'rvec': rvecs[i].squeeze().tolist(),
                'tvec': tvecs[i].squeeze().tolist(),
                'corners': corners.tolist(),
            }
        )
        
    
    with open(os.path.join(output_dir, 'K_D.json'), 'w') as f:
        json.dump(calib_res, f)    
    
    logger.info(f'K:\n{K}')
    logger.info(f'D:\n{D}')
    logger.info(f'Total Mean Error: {retval}')
    

    
def main():
    
    # # 模拟命令行参数
    # args = [
    #     '-i', '/home/frank/data/g3/calib_imgs/fisheyex4/front',
    #     '-c', '/home/frank/data/github/calibtools/calibtools/camera/charuco.yaml',
    #     '-o', '/home/frank/data/github/calibtools/data/calib_results/ffish',
    #     '-s',
    #     '-m', 'fisheye'
    # ]
    # # 替换 sys.argv 的值为模拟的参数列表
    # sys.argv = ['script_name.py'] + args
    
    # # 调用函数
    # calib_camera_charuco()
    
    
    
    # 模拟命令行参数
    args = [
        '-i', '/home/frank/data/g3/calib_imgs/front8m',
        '-c', '/home/frank/data/github/calibtools/calibtools/camera/charuco.yaml',
        '-o', '/home/frank/data/github/calibtools/data/calib_results/f800',
        '-s',
        '-m', 'pinhole'
    ]
    # 替换 sys.argv 的值为模拟的参数列表
    sys.argv = ['script_name.py'] + args
    
    # 调用函数
    calib_camera_charuco()
    
    
    pass
    
    
if __name__ == '__main__':
    main()
    pass
    