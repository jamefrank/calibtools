'''
Author: fanjin 
Date: 2024-06-19 09:57:30
LastEditors: fanjin 
LastEditTime: 2024-06-19 16:30:47
FilePath: /calibtools/python/camera_intrinsic.py
Description: 相机内参标定   phinhole   fisheye

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import cv2
import numpy as np
import glob

# def calib_camera_intrinsic(input_imgs_dir:str, output_res_dir:str):
#     window_name = "calib-intrinsic"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(window_name, 1920, 1080)
    
    
#     cv2.imshow(window_name, result)
#     cv2.waitKey(0)
    
    
#     cv2.destroyAllWindows()
    
#     pass



def read_corners(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    corners = fs.getNode("cornesMat").mat()
    ids = fs.getNode("idsMat").mat()
    fs.release()
    return corners, ids



def main():
    
    camid = 0  # 根据实际情况设置摄像头ID
    output_dir = "/home/frank/data/calib-res-buffer"
    filedir = f"{output_dir}/cam{camid}"  # 输出目录路径
    files = glob.glob(f"{filedir}/*.yml")

    if len(files) == 0:
        return

    image_points = []
    object_points = []
    
    for file in files:
        tmpcorners, tmpids = read_corners(file)
        tmpcorners = tmpcorners.reshape(-1,2)
        # if tmpids.shape[0] != 6:
        #     continue
        im_ps = []
        ob_ps = []
        for j in range(tmpids.shape[0]):
            id = int(tmpids[j, 0])
            idx = id % 3
            idy = id // 3
            ob_ps.append(np.array([(idx + 1) * 0.18, (1 + idy) * 0.18, 0.0],dtype=float))
            im_p = tmpcorners[j, :]
            im_ps.append(im_p)
        image_points.append(np.array(im_ps, dtype=float).reshape(1,-1,2))
        object_points.append(np.array(ob_ps, dtype=float).reshape(1,-1,3))

    # object_points = np.concatenate(object_points, axis=0).reshape(-1,1,3).astype(np.float64)
    # image_points = np.concatenate(image_points, axis=0).reshape(-1,1,2).astype(np.float64)
    # object_points = np.array(object_points).reshape(-1,6,1,3)

    print("calibrating...")
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1024, 1e-9)
    rerror, intrinsic, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
        object_points, image_points, (1280, 720), None, None
        # object_points, image_points, (1280, 720), None, None, None,None, flags, criteria
    )

    print("--- intrinsic ---")
    print(intrinsic)
    print("--- distortion ---")
    print(distortion)
    print("rerror:", rerror)

    print("STEP 2: intrinsic & distortion estimation")
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS | cv2.fisheye.CALIB_FIX_SKEW
    rerror, intrinsic, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
        object_points, image_points, (1280, 720), intrinsic, distortion,None,None, flags, criteria
    )

    print("--- intrinsic ---")
    print(intrinsic)
    print("--- distortion ---")
    print(distortion)
    print("rerror:", rerror)
    
    pass
 
 
if __name__ == '__main__':
    main()
    pass
