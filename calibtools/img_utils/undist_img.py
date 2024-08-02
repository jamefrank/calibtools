'''
Author: fanjin 
Date: 2024-07-19 11:41:28
LastEditors: fanjin 
LastEditTime: 2024-08-02 10:05:22
FilePath: /calibtools/calibtools/img_utils/undist_img.py
Description: 

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import os
import cv2
from typing import List
from tqdm import tqdm
from calibtools.file_utils.time_align import get_files


def patch_undist_img_pinhole(srcdir: str, K, D, output_dir: str):
    files = get_files(srcdir, ".jpg")  # TODO
    abs_paths = [os.path.join(srcdir, path) for path in files]

    pass


def patch_undist_img_pinhole_v1(abs_paths: List[str], K, D, output_dir: str):
    for path in tqdm(abs_paths):
        base_name = os.path.basename(path)
        img = cv2.imread(path)
        img = cv2.undistort(img, K, D)
        cv2.imwrite(os.path.join(output_dir, base_name), img)


def patch_undist_img_fisheye_v1(abs_paths: List[str], K, D, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for path in tqdm(abs_paths):
        base_name = os.path.basename(path)
        img = cv2.imread(path)
        # img = cv2.resize(img, (1920, 1080))
        img = cv2.fisheye.undistortImage(img, K, D, None, K, (img.shape[1], img.shape[0]))
        out_path = os.path.join(output_dir, base_name)
        if out_path == path:
            out_path = os.path.join(output_dir, "un_"+base_name)
        cv2.imwrite(out_path, img)


def patch_undist_img_fisheye(srcdir: str, K, D, output_dir: str):
    files = get_files(srcdir, ".jpg")  # TODO
    abs_paths = [os.path.join(srcdir, path) for path in files]
    patch_undist_img_fisheye_v1(abs_paths, K, D, output_dir)
    pass


def main():
    from calibtools.calib_res_utils.parse import G3_Calib_Parse
    sensor = G3_Calib_Parse()
    # src_dir = "/home/frank/data/G3/scan_fish_calib/fisheyex4/front"
    # output_dir = "/home/frank/data/G3/scan_fish_calib/f_undist"
    # patch_undist_img_fisheye(src_dir, sensor.f_fish.K, sensor.f_fish.D, output_dir)

    src_dir = "/home/frank/data/G3/scan_fish_calib/fisheyex4/back"
    output_dir = "/home/frank/data/G3/scan_fish_calib/b_undist"
    patch_undist_img_fisheye(src_dir, sensor.b_fish.K, sensor.b_fish.D, output_dir)

    src_dir = "/home/frank/data/G3/scan_fish_calib/fisheyex4/left"
    output_dir = "/home/frank/data/G3/scan_fish_calib/l_undist"
    patch_undist_img_fisheye(src_dir, sensor.l_fish.K, sensor.l_fish.D, output_dir)

    src_dir = "/home/frank/data/G3/scan_fish_calib/fisheyex4/right"
    output_dir = "/home/frank/data/G3/scan_fish_calib/r_undist"
    patch_undist_img_fisheye(src_dir, sensor.r_fish.K, sensor.r_fish.D, output_dir)


pass


if __name__ == '__main__':
    main()
    pass
