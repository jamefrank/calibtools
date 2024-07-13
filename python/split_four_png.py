'''
Author: fanjin 
Date: 2024-06-19 18:50:53
LastEditors: fanjin 
LastEditTime: 2024-06-19 19:46:24
FilePath: /calibtools/python/split_four_png.py
Description: 对4张连体图进行切分

l b r f

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import cv2
import numpy as np


def main():
    img_path = "/perception/third_party/g3/20240618/calib_imgs/fisheye_image_yuv_0-1718683250-898701407.png"
    
    save_dir = "/home/frank/data/GitHub/calibtools/data/bev"
    
    img = cv2.imread(img_path)
    W = img.shape[1]//4
    l_img = img[:,:W,:]
    b_img = img[:,W:W*2,:]
    r_img = img[:,W*2:W*3,:]
    f_img = img[:,W*3:,:]
    
    cv2.imwrite(save_dir+"/f.jpg", f_img)
    cv2.imwrite(save_dir+"/b.jpg", b_img)
    cv2.imwrite(save_dir+"/l.jpg", l_img)
    cv2.imwrite(save_dir+"/r.jpg", r_img)

    
    
    print(img.shape)
    pass
 
 
if __name__ == '__main__':
    main()
    pass
