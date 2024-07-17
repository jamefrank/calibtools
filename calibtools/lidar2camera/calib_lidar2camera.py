'''
Author: fanjin 
Date: 2024-07-17 13:46:55
LastEditors: fanjin 
LastEditTime: 2024-07-17 19:41:34
FilePath: /calibtools/calibtools/lidar2camera/calib_lidar2camera.py
Description: targetless calib lidar2camera

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import os
from shlex import join
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
from math import floor
from typing import Tuple
from scipy.spatial import ConvexHull
from calibtools.log import log_utils
from calibtools.camera.common import CamModel
from pc_common.point_cloud.point_cloud import PointCloud
# from calibtools.thirdparty.Super
from super_glue.models.matching import Matching  #TODO
from super_glue.models.utils import frame2tensor, estimate_pose #TODO

logger = log_utils.get_logger() 

def _estimate_camera_fov(image_size: Tuple, K:np.ndarray):
    img_pts = np.array(
        [
            [0,0],
            [image_size[0]/2,0],
            [0,image_size[1]/2]
        ],dtype=float
    )
    
    #
    normalized_pts = np.zeros_like(img_pts)
    normalized_pts[:, 0] = (img_pts[:, 0] - K[0, 2]) / K[0, 0]
    normalized_pts[:, 1] = (img_pts[:, 1] - K[1, 2]) / K[1, 1]
    normalized_pts = np.hstack((normalized_pts, np.ones((normalized_pts.shape[0], 1))))
    
    #
    norms = np.linalg.norm(normalized_pts, axis=1)
    norms += 1e-8
    normalized_pts /= norms[:, np.newaxis]
    fovs = np.arccos(normalized_pts[:, 2])
    fov = np.max(fovs)
    
    return fov

def _estimate_lidar_fov(xyzs: np.ndarray):
    # 计算凸包
    hull = ConvexHull(xyzs)
    
    # 获取凸包顶点坐标
    hull_points = hull.points[hull.vertices]
    
    # 预计算方向向量
    dirs = [point / np.linalg.norm(point) for point in hull_points]
    
    # 找到凸包上的最大角度
    min_cosine = 1.0  # 初始化为1，因为cos(0) = 1，表示最小角度初始化为0度
    for i in range(len(dirs)):
        for j in range(i+1, len(dirs)):
            cosine = np.dot(dirs[i], dirs[j])
            min_cosine = min(cosine, min_cosine)
    
    # 最后，计算最大角度
    max_angle = np.arccos(min_cosine)
    
    return max_angle*1.0
    
# TODO only for static scenes
def _generate_lidar_img(
    pcd_dir: str,
    save_dir: str,
    min_distance:float=1.0,
    voxel_resolution:float=0.001,
    bins:int=256,
    ):
    # 
    logger.info('read img paths...')
    file_extensions = ["*.pcd"]
    pcd_files = []
    for ext in file_extensions:
        pcd_files.extend(glob.glob(os.path.join(pcd_dir, ext)))
        
    #
    voxelgrid = {}
    for lidar_file in pcd_files[:20]:  #TODO for test
    # for lidar_file in pcd_files:
        cloud = PointCloud.read_file(lidar_file)
        selected_fields = ['x', 'y', 'z', 'intensity']
        cloud = PointCloud.from_dict({field: cloud.data[field] for field in selected_fields})
        xyz = cloud.points
        xyz_distance = np.linalg.norm(xyz, axis=1)
        cloud = cloud.select_index(xyz_distance>=min_distance)
        xyz = cloud.points
        xyz_int = np.floor((xyz/voxel_resolution)).astype(int)
        xyzi = cloud.points_with_intensity
        for coord,pt in tqdm(zip(xyz_int,xyzi), total=len(xyz_int)):
            voxelgrid[tuple(coord)] = pt
    
    
    # 
    logger.info('histrogram equalization...') 
    pts = [value for value in voxelgrid.values()]
    pts = np.array(pts)
    intensity_index = -1
    sorted_indices = np.argsort(pts[:, intensity_index])
    for i,indice in enumerate(sorted_indices):
        pts[indice, intensity_index] = floor(i*bins/len(sorted_indices))
        # pts[indice, intensity_index] = floor(i*bins/len(sorted_indices))/bins
    
    lidar_fov = _estimate_lidar_fov(pts[:,:3])
    logger.info(f'lidar fov: {np.rad2deg(lidar_fov)} [deg]')
    lidar_image_size = (1024, 1024)
    lidar_image_size = (1080, 1080)
    fx = lidar_image_size[0]/2/np.tan(lidar_fov/2)
    lidar_K = np.array(
        [
            [fx, 0, lidar_image_size[0]/2],
            [0, fx, lidar_image_size[1]/2],
            [0, 0, 1]
        ],dtype=float
    )
    # TODO
    # lidar: 前左上
    # camera: 右下前
    T_camera_lidar =np.array(
        [
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ],dtype=float
    )
    R_camera_lidar = T_camera_lidar[:3,:3]
    t_camera_lidar = T_camera_lidar[:3,3]
    
    #
    camera_fov = _estimate_camera_fov(lidar_image_size, lidar_K)
    min_z = 1.0 * np.cos(camera_fov)
    
    intensity_image = np.zeros(lidar_image_size, dtype=np.float64)
    index_image = np.full(lidar_image_size, -1, dtype=np.int32)
    
    pts_camera = (np.dot(R_camera_lidar, pts[:,:3].T) + t_camera_lidar.reshape((3,1))).T
    norms = np.linalg.norm(pts_camera, axis=1)
    norms += 1e-8
    pts_camera_normalized = pts_camera/norms[:, np.newaxis]

    
    pts_img = (lidar_K@pts_camera.T).T
    pts_img = pts_img / pts_img[:,2][:, np.newaxis]
    pts_img = pts_img.astype(int)
    pts_img = pts_img[:,:2]
    
    selected_indices = (
        (pts_camera_normalized[:, 2] >= min_z) &
        (pts_img[:, 0] >= 0) &
        (pts_img[:, 0] < lidar_image_size[0]) &
        (pts_img[:, 1] >= 0) &
        (pts_img[:, 1] < lidar_image_size[1])
    )
    true_indices = np.where(selected_indices)[0]
    
    pts_img = pts_img[selected_indices]
    selected_pts = pts[selected_indices]
    
    intensity_image[pts_img[:,1], pts_img[:,0]] = selected_pts[:,3] 
    index_image[pts_img[:,1], pts_img[:,0]] = true_indices
    
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, 'lidar_intensity.png'), intensity_image)
    # cv2.imwrite(os.path.join(save_dir, 'lidar_index.png'), index_image)
    
    np.save(os.path.join(save_dir, 'lidar_index.npy'), index_image)
    np.save(os.path.join(save_dir, 'lidar_pts.npy'), pts)
    
    # return intensity_image, index_image
    pass


def _generate_camera_img(
    image_file: str, 
    K:np.ndarray, 
    D:np.ndarray,
    output_dir: str
    ):
    img = cv2.imread(image_file)
    img = cv2.undistort(img, K, D, None, K)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(img, img)
    
    cv2.imwrite(os.path.join(output_dir, 'camera_img.png'), img)
    

def _init_calib_lidar_to_camera(
    lidar_img_file: str,
    camera_img_file: str,
    lidar_indieces_file: str,
    lidar_pts_file: str
):
    torch.set_grad_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval().to(device)

    #
    image1 = cv2.imread(str(lidar_img_file), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(str(camera_img_file), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.resize(image2, (1920, 1080))
    inp0 = frame2tensor(image1, device)
    inp1 = frame2tensor(image2, device)

    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
   
    if len(mkpts0) == 0:
        logger.info("num of pts is 0, calib failed!")
        return
    else:
        image1 = cv2.imread(str(lidar_img_file))
        image2 = cv2.imread(str(camera_img_file))
        image2 = cv2.resize(image2, (1920, 1080))
        result = np.hstack((image2, image1))
        for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
            x0 = int(x0)
            x1 = int(x1)
            y0 = int(y0)
            y1 = int(y1)
            cv2.line(result, (x1, y1), (x0 + 1920, y0), color=tuple(np.random.randint(0, 255, 3).tolist()), thickness=1, lineType=cv2.LINE_AA)
    
    #     window_name = 'superglue'
    #     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #     cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #     cv2.resizeWindow(window_name, 1920, 1080)
    #     cv2.imshow(window_name, result)
    #     cv2.setWindowTitle(window_name, str(len(mkpts0)))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    
    #
    pick_window_size = 1
    range_array = np.arange(-pick_window_size, pick_window_size + 1)
    x, y = np.meshgrid(range_array, range_array)
    pick_offsets = np.vstack([x.flatten(), y.flatten()]).T
    
    #
    lidar_index_img = np.load(lidar_indieces_file)
    lidar_pts = np.load(lidar_pts_file)
    correspondences = []
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        x0 = int(x0)
        y0 = int(y0)
        point_index = lidar_index_img[x0, y0]
        if point_index < 0:
            for offset in pick_offsets:
                point_index = lidar_index_img[x0 + offset[0], y0 + offset[1]]
                if point_index >= 0:
                    break
            if point_index < 0:
                continue
        correspondences.append((lidar_pts[point_index,:3], np.array((x1, y1),dtype=float)))
    logger.info(f'matches:{len(mkpts0)}')
    logger.info(f'correspondences:{len(correspondences)}')
    
    if len(correspondences) < 2:
        logger.info("num of correspondences is <2, calib failed!")
        return
    
    
    
    pass


def main():
    pcd_dir = "/home/frank/data/G3/at128_cam800/calib_floor.parsed/sensor/lidar_front_at128"
    output_dir = "/home/frank/data/GitHub/calibtools/calibtools/lidar2camera/data"
    _generate_lidar_img(pcd_dir, output_dir)
    
    # K = np.array(
    #     [
    #         [2480.296422347829, 0, 1900.995348696508],
    #         [0, 2483.750770053912, 1103.437012143754],
    #         [0, 0, 1]
    #     ],dtype=float
    # )
    # D = np.array([
    #   5.382911218057614,
    #   0.8531957591786572,
    #   -0.0002046633281157292,
    #   0.0001862206769627878,
    #   -0.180148309027261,
    #   6.117056505700593,
    #   4.000238859043452,
    #   -0.401455901730928,
    # ])
    # img_file = "/home/frank/data/GitHub/calibtools/calibtools/lidar2camera/data/1719817366.238261657.jpg"
    # _generate_camera_img(img_file, K, D, output_dir)
    
    
    # #
    lidar_img_file = "/home/frank/data/GitHub/calibtools/calibtools/lidar2camera/data/lidar_intensity.png"
    camera_img_file = "/home/frank/data/GitHub/calibtools/calibtools/lidar2camera/data/camera_img.png"
    lidar_index_img_file = "/home/frank/data/GitHub/calibtools/calibtools/lidar2camera/data/lidar_index.npy"
    lidar_pts_file = "/home/frank/data/GitHub/calibtools/calibtools/lidar2camera/data/lidar_pts.npy"
    _init_calib_lidar_to_camera(lidar_img_file, camera_img_file, lidar_index_img_file, lidar_pts_file)
    pass
 
 
if __name__ == '__main__':
    main()
    pass



