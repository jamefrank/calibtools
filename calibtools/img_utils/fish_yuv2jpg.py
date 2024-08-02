import os
import sys

import shutil
import subprocess

import numpy as np
import cv2

# depend on ffmpeg


def is_front8m(img_path: str) -> bool:
    yuv = np.fromfile(img_path, dtype=np.uint8)
    front8m_shape = (3840, 2160, 2)
    fisheyex4_shape = (5120, 720, 2)
    if yuv.shape[0] == front8m_shape[0] * front8m_shape[1] * front8m_shape[2]:
        return True
    elif yuv.shape[0] == 5120 * 720 * 2:
        return False
    else:
        raise ValueError('Invalid image shape')


def new_jpg_path(img_path: str, is_front8m: bool) -> str:
    file_name = os.path.basename(img_path)
    file_name = os.path.splitext(file_name)[0]
    par_dir = os.path.dirname(os.path.dirname(img_path))
    if is_front8m:
        os.makedirs(os.path.join(par_dir, 'front8m'), exist_ok=True)
        jpg_path = os.path.join(par_dir, 'front8m', file_name + '.jpg')
    else:
        os.makedirs(os.path.join(par_dir, 'fisheyex4'), exist_ok=True)
        jpg_path = os.path.join(par_dir, 'fisheyex4', file_name + '.jpg')
    return jpg_path


def yuv2rgb_command(img_path: str, is_front8m: bool) -> str:
    jpg_path = new_jpg_path(img_path, is_front8m)
    if is_front8m:
        size_str = '3840x2160'
    else:
        size_str = '5120x720'
    command = f'ffmpeg -y -s {size_str} -pix_fmt yuv422p -i {img_path} -q:v 1 {jpg_path}'
    return command, jpg_path


def scan_dir(dir_path: str) -> None:
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.yuv'):
                file_path = os.path.join(root, file)
                flag = is_front8m(file_path)
                command, jpg_path = yuv2rgb_command(file_path, flag)
                print(f'{command=}')
                subprocess.run(command, shell=True)
                if not flag:
                    split_fisheyex4(jpg_path)


def split_fisheyex4(jpg_path: str) -> None:
    order = ['front', 'left', 'right', 'back']
    dir_path = os.path.dirname(jpg_path)
    for o in order:
        os.makedirs(os.path.join(dir_path, o), exist_ok=True)
    img = cv2.imread(jpg_path)
    h, w, _ = img.shape
    w = w // 4
    for i, o in enumerate(order):
        x1 = i * w
        x2 = (i + 1) * w
        img_crop = img[:, x1:x2]
        cv2.imwrite(os.path.join(dir_path, o, os.path.basename(jpg_path)), img_crop)


def main():
    if len(sys.argv) != 2:
        print('Usage: python fish_yuv2jpg.py <dir_path>')
        sys.exit(1)
    dir_path = sys.argv[1]
    scan_dir(dir_path)


if __name__ == '__main__':
    main()
