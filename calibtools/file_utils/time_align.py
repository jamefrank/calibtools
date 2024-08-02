'''
Author: fanjin 
Date: 2024-07-19 11:18:17
LastEditors: fanjin 
LastEditTime: 2024-08-02 10:34:57
FilePath: /calibtools/calibtools/file_utils/time_align.py
Description: 查找文件时间戳对齐 pairs

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import os
from torch import abs_
from tqdm import tqdm
from typing import List
import numpy as np


def get_files(folder_path, ext):
    jpg_files = []
    for file_name in os.listdir(folder_path):
        if not file_name.startswith('.') and file_name.endswith(ext):
            jpg_files.append(file_name)
    sorted_jpg_files = sorted(jpg_files, key=lambda x: x.lower())
    return sorted_jpg_files


def get_time_align_pairs(f_dir: str, other_dirs: List[str], ts_sub_strs: List[str], thresh: float = 25*0.001):
    f_files = get_files(f_dir, ts_sub_strs[0])
    other_files = [get_files(item, ts_sub_strs[idx+1]) for idx, item in enumerate(other_dirs)]
    other_nums = len(other_dirs)
    other_idx = [0]*other_nums
    other_min_dts = [np.inf]*other_nums

    def file_ts(f_file, ts_sub_str): return float(os.path.basename(f_file).replace(ts_sub_str, ""))

    res_pairs = []

    # b_found_finish = False
    b_found_finish = True
    for f_file in tqdm(f_files):
        f_ts = file_ts(f_file, ts_sub_strs[0])
        #
        for i in range(other_nums):
            while other_idx[i] <= len(other_files[i])-1 and abs(file_ts(other_files[i][other_idx[i]], ts_sub_strs[i+1])-f_ts) < other_min_dts[i]:
                other_min_dts[i] = abs(file_ts(other_files[i][other_idx[i]], ts_sub_strs[i+1])-f_ts)
                other_idx[i] += 1

        #
        if max(other_min_dts) < thresh:
            pairs = [files[idx-1] for idx, files in zip(other_idx, other_files)]
            pairs.insert(0, f_file)
            res_pairs.append(pairs)

        #
        for i in range(other_nums):
            b_found_finish = b_found_finish and other_idx[i] == len(other_files[i])
        if b_found_finish:
            break
        #
        other_min_dts = [np.inf]*other_nums
        other_idx = [idx-1 for idx in other_idx]

    return res_pairs


def ln(src_dirs: List[str], target_dir: str):
    import os

    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有文件
    for src_dir in src_dirs:
        for filename in os.listdir(src_dir):
            if filename.endswith('.pcd'):
                # 构建完整的源文件路径
                src_file_path = os.path.join(src_dir, filename)

                # 构建目标文件路径
                target_file_path = os.path.join(target_dir, filename)

                # 创建软链接
                try:
                    os.symlink(src_file_path, target_file_path)
                    print(f'Successfully created symlink for {filename}')
                except OSError as e:
                    print(f'Failed to create symlink for {filename}: {e}')
    pass


def ln_v2(src_dirs: List[str], sub_names: List[str], target_dir: str):
    import os

    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有文件
    for idx, src_dir in enumerate(src_dirs):
        for filename in os.listdir(src_dir):
            if filename.endswith('.pcd'):
                # if filename.endswith('.jpg'):
                # 构建完整的源文件路径
                src_file_path = os.path.join(src_dir, filename)

                # 构建目标文件路径
                target_file_path = os.path.join(
                    target_dir, sub_names[idx]+"_"+filename) if len(sub_names) == len(src_dirs) else os.path.join(target_dir, filename)

                # 创建软链接
                try:
                    os.symlink(src_file_path, target_file_path)
                    print(f'Successfully created symlink for {filename}')
                except OSError as e:
                    print(f'Failed to create symlink for {filename}: {e}')
    pass


def ln_files(abs_paths: List[str], target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    for path in abs_paths:
        basename = os.path.basename(path)
        target_file_path = os.path.join(target_dir, basename)
        # 创建软链接
        try:
            os.symlink(path, target_file_path)
            # print(f'Successfully created symlink for {basename}')
        except OSError as e:
            print(f'Failed to create symlink for {basename}: {e}')
    pass


def main():
    # # 所有的pcd集中到一个文件夹内
    # dst_dir = "/home/frank/data/G3/scan_fish_calib/all_pcds"
    # src_dirs = [
    #     "/home/frank/data/G3/scan_fish_calib/rslidar/rslidar_2024-08-01-16-31-52_0/sensor/top_st_m1",
    #     "/home/frank/data/G3/scan_fish_calib/rslidar/rslidar_2024-08-01-16-36-52_1/sensor/top_st_m1",
    #     "/home/frank/data/G3/scan_fish_calib/rslidar/rslidar_2024-08-01-16-41-52_2/sensor/top_st_m1"
    # ]
    # sub_names = []
    # ln_v2(src_dirs, sub_names, dst_dir)

    # 获取文件时间戳对齐pairs
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

    ln_files(abs_pcd, "/home/frank/data/G3/scan_fish_calib/pcd_f_align")
    ln_files(abs_f, "/home/frank/data/G3/scan_fish_calib/pcd_f_align")

    ln_files(abs_pcd, "/home/frank/data/G3/scan_fish_calib/pcd_b_align")
    ln_files(abs_b, "/home/frank/data/G3/scan_fish_calib/pcd_b_align")

    ln_files(abs_pcd, "/home/frank/data/G3/scan_fish_calib/pcd_l_align")
    ln_files(abs_l, "/home/frank/data/G3/scan_fish_calib/pcd_l_align")

    ln_files(abs_pcd, "/home/frank/data/G3/scan_fish_calib/pcd_r_align")
    ln_files(abs_r, "/home/frank/data/G3/scan_fish_calib/pcd_r_align")

    pass


if __name__ == '__main__':
    main()
    pass
