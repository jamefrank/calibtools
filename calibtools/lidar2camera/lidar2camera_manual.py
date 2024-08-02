'''
Author: fanjin 
Date: 2024-07-19 11:39:42
LastEditors: fanjin 
LastEditTime: 2024-08-02 14:18:56
FilePath: /calibtools/calibtools/lidar2camera/lidar2camera_manual.py
Description: 

conclusion:
1. 机械lidar线束太稀疏,提取点的精度太差，采用面特征会精度更高
2. 使用map点云能增强点的测量精度,标定精度会更高，但是难度受限于建图

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List
from calibtools.file_utils.time_align import get_time_align_pairs, ln_files
from calibtools.img_utils.undist_img import patch_undist_img_pinhole_v1
from calibtools.file_utils.time_align import ln
from calibtools.calib_res_utils.parse import G3_Calib_Parse
from calibtools.opt_utils.opt_3d_2d import calib_3d_to_2d
from calibtools.img_utils.lidar_project_img import project_pcd_to_img


def _prepare_data(
    src_img_dir: str,
    src_pcd_dir: str,
    K,
    D,
    output_dir: str,
    ts_sub_strs: List[str] = [".jpg.jpg", ".pcd"]
):
    os.makedirs(output_dir, exist_ok=True)
    #
    # ts_sub_strs = [".jpg", ".pcd"]
    pairs = get_time_align_pairs(src_img_dir, [src_pcd_dir], ts_sub_strs)

    #
    img_abs_paths = [os.path.join(src_img_dir, pair[0]) for pair in pairs]
    patch_undist_img_pinhole_v1(img_abs_paths, K, D, output_dir)

    #
    pcd_abs_paths = [os.path.join(src_pcd_dir, pair[1]) for pair in pairs]
    ln_files(pcd_abs_paths, output_dir)

    pass


def preprocess():
    sensor = G3_Calib_Parse()
    #
    src_pcd_dir = "/home/frank/data/G3/scan_calib/20240718/ros1/all_pcds"

    #
    output_dir = "/home/frank/data/G3/scan_calib/20240718/scan_calib_lb"
    src_img_dir = "/home/frank/data/GitHub/calibtools/data/calib_results/side-lb/temp"
    _prepare_data(src_img_dir, src_pcd_dir, sensor.scan_lb.K, sensor.scan_lb.D, output_dir)

    #
    output_dir = "/home/frank/data/G3/scan_calib/20240718/scan_calib_lf"
    src_img_dir = "/home/frank/data/GitHub/calibtools/data/calib_results/side-lf/temp"
    _prepare_data(src_img_dir, src_pcd_dir, sensor.scan_lf.K, sensor.scan_lf.D, output_dir)

    #
    output_dir = "/home/frank/data/G3/scan_calib/20240718/scan_calib_rb"
    src_img_dir = "/home/frank/data/GitHub/calibtools/data/calib_results/side-rb/temp"
    _prepare_data(src_img_dir, src_pcd_dir, sensor.scan_rb.K, sensor.scan_rb.D, output_dir)

    #
    output_dir = "/home/frank/data/G3/scan_calib/20240718/scan_calib_rf"
    src_img_dir = "/home/frank/data/GitHub/calibtools/data/calib_results/side-rf/temp"
    _prepare_data(src_img_dir, src_pcd_dir, sensor.scan_rf.K, sensor.scan_rf.D, output_dir)

    #
    output_dir = "/home/frank/data/G3/scan_calib/20240718/scan_calib_f"
    src_img_dir = "/home/frank/data/GitHub/calibtools/data/calib_results/side-f/temp"
    _prepare_data(src_img_dir, src_pcd_dir, sensor.scan_f.K, sensor.scan_f.D, output_dir)

    #
    output_dir = "/home/frank/data/G3/scan_calib/20240718/scan_calib_b"
    src_img_dir = "/home/frank/data/GitHub/calibtools/data/calib_results/side-b/temp"
    _prepare_data(src_img_dir, src_pcd_dir, sensor.scan_b.K, sensor.scan_b.D, output_dir)

    pass


def calib_init_guess(pcd_yml: str, img_yml: str, K):
    fs = cv2.FileStorage(pcd_yml, cv2.FileStorage_READ)
    pcd_corners = fs.getNode("corners").mat()

    fs = cv2.FileStorage(img_yml, cv2.FileStorage_READ)
    img_corners = fs.getNode("corners").mat()

    T = calib_3d_to_2d(pcd_corners, img_corners, K)

    return T


def main():
    # preprocess()

    sensor = G3_Calib_Parse()

    pcd_dir = "/home/frank/data/G3/scan_fish_calib/all_pcds"
    f_dir = "/home/frank/data/G3/scan_fish_calib/f_undist"
    b_dir = "/home/frank/data/G3/scan_fish_calib/b_undist"
    l_dir = "/home/frank/data/G3/scan_fish_calib/l_undist"
    r_dir = "/home/frank/data/G3/scan_fish_calib/r_undist"

    pairs = get_time_align_pairs(pcd_dir, [f_dir, b_dir, l_dir, r_dir], [".pcd", ".jpg", ".jpg", ".jpg", ".jpg"], thresh=0.1*0.5)

    abs_pcd = [os.path.join(pcd_dir, item[0]) for item in pairs]
    abs_f = [os.path.join(f_dir, item[1]) for item in pairs]
    abs_b = [os.path.join(b_dir, item[2]) for item in pairs]
    abs_l = [os.path.join(l_dir, item[3]) for item in pairs]
    abs_r = [os.path.join(r_dir, item[4]) for item in pairs]

    # #
    # calib_dir = "/home/frank/data/G3/scan_fish_calib/calib_f"
    # pcd_yml = calib_dir + "/pcd_corners.yaml"
    # img_yml = calib_dir + "/img_corners.yaml"
    # lidar2camera = calib_init_guess(pcd_yml, img_yml, sensor.f_fish.K)
    # output_dir = calib_dir
    # for idx, (img_path, pcd_path) in tqdm(enumerate(zip(abs_f, abs_pcd))):
    #     project_pcd_to_img(img_path, pcd_path, lidar2camera, sensor.f_fish.K, output_dir)
    # pass

    # #
    # calib_dir = "/home/frank/data/G3/scan_fish_calib/calib_b"
    # pcd_yml = calib_dir + "/pcd_corners.yaml"
    # img_yml = calib_dir + "/img_corners.yaml"
    # lidar2camera = calib_init_guess(pcd_yml, img_yml, sensor.b_fish.K)
    # output_dir = calib_dir
    # for idx, (img_path, pcd_path) in tqdm(enumerate(zip(abs_b, abs_pcd))):
    #     if idx < len(abs_b)-50:
    #         continue
    #     project_pcd_to_img(img_path, pcd_path, lidar2camera, sensor.b_fish.K, output_dir)
    # pass

    #
    # calib_dir = "/home/frank/data/G3/scan_fish_calib/calib_l"
    # pcd_yml = calib_dir + "/pcd_corners.yaml"
    # img_yml = calib_dir + "/img_corners.yaml"
    # lidar2camera = calib_init_guess(pcd_yml, img_yml, sensor.l_fish.K)
    # output_dir = calib_dir
    # for idx, (img_path, pcd_path) in tqdm(enumerate(zip(abs_l, abs_pcd))):
    #     if idx < len(abs_b)-50:
    #         continue
    #     project_pcd_to_img(img_path, pcd_path, lidar2camera, sensor.l_fish.K, output_dir)
    # pass

    calib_dir = "/home/frank/data/G3/scan_fish_calib/calib_r"
    pcd_yml = calib_dir + "/pcd_corners.yaml"
    img_yml = calib_dir + "/img_corners.yaml"
    lidar2camera = calib_init_guess(pcd_yml, img_yml, sensor.r_fish.K)
    output_dir = calib_dir
    for idx, (img_path, pcd_path) in tqdm(enumerate(zip(abs_r, abs_pcd))):
        if idx < len(abs_b)-300:
            continue
        project_pcd_to_img(img_path, pcd_path, lidar2camera, sensor.r_fish.K, output_dir)
    pass


if __name__ == '__main__':
    main()
    pass
