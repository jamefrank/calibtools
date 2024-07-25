'''
Author: fanjin 
Date: 2024-07-19 11:18:17
LastEditors: fanjin 
LastEditTime: 2024-07-20 22:37:55
FilePath: /calibtools/calibtools/file_utils/time_align.py
Description: 查找文件时间戳对齐 pairs

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import os
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
    for src_dir, sub_name in zip(src_dirs, sub_names):
        for filename in os.listdir(src_dir):
            # if filename.endswith('.pcd'):
            if filename.endswith('.jpg'):
                # 构建完整的源文件路径
                src_file_path = os.path.join(src_dir, filename)

                # 构建目标文件路径
                target_file_path = os.path.join(target_dir, sub_name+"_"+filename)

                # 创建软链接
                try:
                    os.symlink(src_file_path, target_file_path)
                    print(f'Successfully created symlink for {filename}')
                except OSError as e:
                    print(f'Failed to create symlink for {filename}: {e}')
    pass


def ln_files(abs_paths: List[str], target_dir: str):
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
    # #
    # dst_dir = "/home/frank/data/G3/scan_calib/20240718/lb-rf"
    # src_dirs = [
    #     "/home/frank/data/G3/scan_calib/20240718/side4/lb",
    #     "/home/frank/data/G3/scan_calib/20240718/side4/rf"
    # ]
    # sub_names = ["lb", "rf"]
    # ln_v2(src_dirs, sub_names, dst_dir)

    #
    dst_dir = "/home/frank/data/G3/scan_calib/20240718/lf-rb"
    src_dirs = [
        "/home/frank/data/G3/scan_calib/20240718/side4/lf",
        "/home/frank/data/G3/scan_calib/20240718/side4/rb"
    ]
    sub_names = ["lf", "rb"]
    ln_v2(src_dirs, sub_names, dst_dir)

    pass


if __name__ == '__main__':
    main()
    pass
