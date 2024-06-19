# -*- encoding: utf-8 -*-
#@File    :   calib.py
#@Time    :   2024/04/09 15:22:51
#@Author  :   frank 
#@Email:
#@Description: 读取标定参数的工具函数

import cv2
import math
import copy
import yaml
import numpy as np
from enum import Enum
from pathlib import Path
from dataclasses import dataclass 

#
class CamModel(Enum):
    PINHOLE = 1
    FISHEYE = 2

class BevFilterType(Enum):
    FilterF = 1
    FilterB = 2
    FilterL = 3
    FilterR = 4

@dataclass
class CamParam:
    K: np.ndarray = None
    D: np.ndarray = None
    theta: np.ndarray = None
    r_distorted: np.ndarray = None
    ext_car2cam: np.ndarray = None
    H_cam2bev: np.ndarray = None
    newFxScale: float = 1.0 
    newFyScale: float = 1.0
    img_h: int = None
    img_w: int = None
    model: CamModel=CamModel.PINHOLE
    bev_filter_type:BevFilterType=BevFilterType.FilterB

@dataclass
class BevInsParam:
    newFxScale: float = None 
    newFyScale: float = None
    f_dis: float = None
    b_dis: float = None
    l_dis: float = None
    r_dis: float = None
    resolution: float = None
    offx: int = 0
    offy: int = 0
    bev_width: int = None
    bev_height: int = None
    # 虚拟相机内参
    virtual_K: np.ndarray=None


@dataclass
class SensorConfig:
    f_fish: CamParam = CamParam()
    b_fish: CamParam = CamParam()
    l_fish: CamParam = CamParam()
    r_fish: CamParam = CamParam()
    front800: CamParam = CamParam()
    front200: CamParam = CamParam()
    back200: CamParam = CamParam()
    bev: BevInsParam = BevInsParam()
    at128To800: np.ndarray=None
    backRslidarToback200: np.ndarray=None
  
def calc_bev_K(bev:BevInsParam):
    virtual_fx = -1/bev.resolution
    scale_x = bev.l_dis/(bev.l_dis + bev.r_dis)
    virtual_cx = bev.bev_width*scale_x + bev.offx
    
    virtual_fy = -1/bev.resolution
    scale_y = bev.f_dis/(bev.f_dis + bev.b_dis)
    virtual_cy = bev.bev_height*scale_y + bev.offy
    
    virtual_K = np.array(
        [
            [0, virtual_fx, virtual_cx],
            [virtual_fy, 0, virtual_cy],
            [0, 0, 1]
        ],dtype=np.float64
    )
    
    return virtual_K

def calc_H(ext:np.ndarray, K:np.ndarray, K_bev:np.ndarray, newFxScale:float=1, newFyScale:float=1):
    """计算相机到bev的单应性矩阵

    Args:
        ext (np.ndarray): 车辆到相机的外参
        K (np.ndarray): 相机内参
        K_bev (np.ndarray): 车辆到bev的虚拟内参
    """
    newK = copy.deepcopy(K)
    newK[0,0] *= newFxScale
    newK[1,1] *= newFyScale
    
    T = ext[:3,[0,1,3]]
    KT = newK@T
    H = K_bev@np.linalg.inv(KT)
    H = H / H[2,2]
    # print(H)
    return H

def fish_distorted_table(distcoef: np.ndarray):
    '''
    鱼眼相机畸变表
    '''
    # theta
    angle = np.arange(start=0, stop=90+0.1, step=0.1, dtype=np.float64)
    angle_radian = angle*np.pi/180
    r_distorted = angle_radian \
        + distcoef[0]*angle_radian**3 \
            + distcoef[1]*angle_radian**5 \
                + distcoef[2]*angle_radian**7 \
                    + distcoef[3]*angle_radian**9
    return angle_radian,r_distorted

def binaryFind(r_distorted,angle_radian_table,r_distorted_table):
    if r_distorted < r_distorted_table[0] or r_distorted >= r_distorted_table[-1]:
        return False,None
    left_idx = 0
    right_idx = len(r_distorted_table)-1
    while right_idx - left_idx > 1:
        if r_distorted_table[left_idx]==r_distorted:
            return True,angle_radian_table[left_idx]
        if r_distorted_table[right_idx]==r_distorted:
            return True,angle_radian_table[right_idx]
        mid_idx = int((left_idx+right_idx)*0.5)
        if r_distorted_table[mid_idx]==r_distorted:
            return True,angle_radian_table[mid_idx]
        elif r_distorted_table[mid_idx]<r_distorted:
            left_idx = mid_idx
        elif r_distorted_table[mid_idx]>r_distorted:
            right_idx = mid_idx
    return True,(r_distorted-r_distorted_table[left_idx])/(r_distorted_table[right_idx]-r_distorted_table[left_idx])*(angle_radian_table[right_idx]-angle_radian_table[left_idx]) + angle_radian_table[left_idx]

def fish_undistort_img(img, K, D, scalex=1, scaley=1, imshow=False):

    Knew = K.copy()
    if scalex:  # change fov
        Knew[0, 0] *= scalex
    if scaley:
        Knew[1, 1] *= scaley
    DIM = [img.shape[1], img.shape[0]]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    if imshow:
        cv2.namedWindow("undist", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("undist", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.setMouseCallback('undist', mouse_callback)
        cv2.resizeWindow("undist", 1280, 720)
        cv2.imshow("undist", undistorted_img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return undistorted_img

def fish_undistort_points(src_points, angle_radian_table, r_distorted_table, K, newK):
    """ 鱼眼相机畸变点去畸变

    Args:
        src_points (_type_): N*2
        angle_radian_table (_type_): 折射角
        r_distorted_table (_type_): 畸变半径
        K (_type_): 相机内参
        newK (_type_): 去畸变所在的相机内参

    Raises:
        ValueError: _description_

    Returns:
        _type_: N*2
    """
    if len(src_points) == 0:
        return []
    dst = []
    for p in src_points:
        # distorted to normalized plane
        x_normalized = (p[0] - K[0, 2]) / K[0, 0]
        y_normalized = (p[1] - K[1, 2]) / K[1, 1]

        # d_distorted
        r_distorted = math.sqrt(pow(x_normalized, 2) + pow(y_normalized, 2))
        bfind,theta = binaryFind(r_distorted,angle_radian_table,r_distorted_table)
        # 
        if bfind:
            r_undistorted = math.tan(theta)
            x_normalized_undist = x_normalized * r_undistorted / r_distorted
            y_normalized_undist = y_normalized * r_undistorted / r_distorted

            xy = np.array(
                [
                    x_normalized_undist * newK[0, 0] + newK[0, 2],
                    y_normalized_undist * newK[1, 1] + newK[1, 2],
                ],
                dtype=np.float32,
            )

            dst.append(xy)
        else:
            # raise ValueError("not find angle_of_incidence")
            continue
    return np.array(dst)

def fish_undistort_points_FxyScale(src_points, angle_radian_table, r_distorted_table, K, newFxScale:float=1, newFyScale:float=1):
    """同 fish_undistort_points

    Args:
        src_points (_type_): _description_
        angle_radian_table (_type_): _description_
        r_distorted_table (_type_): _description_
        K (_type_): _description_
        newFxScale (float, optional): _description_. Defaults to 1.
        newFyScale (float, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    newK = copy.deepcopy(K)
    newK[0,0] *= newFxScale
    newK[1,1] *= newFyScale
    return fish_undistort_points(src_points,angle_radian_table,r_distorted_table,K,newK)

def undisPoints2bev(src_points, H):  
    """去畸变点转到bev下

    Args:
        src_points (_type_): N*2
        H (_type_): 单应性矩阵

    Returns:
        _type_: N*2
    """
    src_points_homogeneous = np.hstack((src_points, np.ones((src_points.shape[0], 1))))
    bev_points = (H@src_points_homogeneous.T).T
    bev_points = bev_points[:, :2] / bev_points[:,2:]
    return bev_points[:,:2]

def Calib_Parse():
    """解析传感器的内外参
    """
    sensor = SensorConfig()
    calib_file = Path(__file__).parent/"calib.yaml"
    with open(calib_file, 'r') as file:
        config = yaml.safe_load(file)
        # 解析相机参数
        sensor.f_fish.K = np.array(config['FrontFish']['K'], dtype=np.float64)
        sensor.f_fish.D = np.array(config['FrontFish']['D'], dtype=np.float64)
        sensor.f_fish.theta, sensor.f_fish.r_distorted = fish_distorted_table(sensor.f_fish.D)
        sensor.f_fish.ext_car2cam = np.array(config['TGroundFrontFish'], dtype=np.float64)
        sensor.f_fish.img_h = 720
        sensor.f_fish.img_w = 1280
        sensor.f_fish.model = CamModel.FISHEYE
        sensor.f_fish.bev_filter_type = BevFilterType.FilterB
        
        sensor.b_fish.K = np.array(config['BackFish']['K'], dtype=np.float64)
        sensor.b_fish.D = np.array(config['BackFish']['D'], dtype=np.float64)
        sensor.b_fish.theta, sensor.b_fish.r_distorted = fish_distorted_table(sensor.b_fish.D)
        sensor.b_fish.ext_car2cam = np.array(config['TGroundBackFish'], dtype=np.float64)
        sensor.b_fish.img_h = 720
        sensor.b_fish.img_w = 1280
        sensor.b_fish.model = CamModel.FISHEYE
        sensor.b_fish.bev_filter_type = BevFilterType.FilterF

        sensor.l_fish.K = np.array(config['LeftFish']['K'], dtype=np.float64)
        sensor.l_fish.D = np.array(config['LeftFish']['D'], dtype=np.float64)
        sensor.l_fish.theta, sensor.l_fish.r_distorted = fish_distorted_table(sensor.l_fish.D)
        sensor.l_fish.ext_car2cam = np.array(config['TGroundLeftFish'], dtype=np.float64)
        sensor.l_fish.img_h = 720
        sensor.l_fish.img_w = 1280
        sensor.l_fish.model = CamModel.FISHEYE
        sensor.l_fish.bev_filter_type = BevFilterType.FilterR

        sensor.r_fish.K = np.array(config['RightFish']['K'], dtype=np.float64)
        sensor.r_fish.D = np.array(config['RightFish']['D'], dtype=np.float64)
        sensor.r_fish.theta, sensor.r_fish.r_distorted = fish_distorted_table(sensor.r_fish.D)
        sensor.r_fish.ext_car2cam = np.array(config['TGroundRightFish'], dtype=np.float64)
        sensor.r_fish.img_h = 720
        sensor.r_fish.img_w = 1280
        sensor.r_fish.model = CamModel.FISHEYE
        sensor.r_fish.bev_filter_type = BevFilterType.FilterL

        sensor.front800.K = np.array(config['Front800']['K'], dtype=np.float64)
        sensor.front800.D = np.array(config['Front800']['D'], dtype=np.float64)
        sensor.front800.ext_car2cam = np.array(config['TGround800'], dtype=np.float64)
        sensor.front800.img_h = 2160
        sensor.front800.img_w = 3840
        
        # bev 参数
        sensor.bev.newFxScale = config['newFxScale']
        sensor.bev.newFyScale = config['newFyScale']
        sensor.bev.f_dis = config['f_dis']
        sensor.bev.b_dis = config['b_dis']
        sensor.bev.l_dis = config['l_dis']
        sensor.bev.r_dis = config['r_dis']
        sensor.bev.resolution = config['resolution']
        sensor.bev.offx = config['offx']
        sensor.bev.offy = config['offy']
        sensor.bev.bev_width = int((sensor.bev.l_dis+sensor.bev.r_dis)/sensor.bev.resolution)
        sensor.bev.bev_height= int((sensor.bev.f_dis+sensor.bev.b_dis)/sensor.bev.resolution)        
        sensor.bev.virtual_K = calc_bev_K(sensor.bev)
        
        #
        sensor.f_fish.H_cam2bev = calc_H(sensor.f_fish.ext_car2cam, sensor.f_fish.K, sensor.bev.virtual_K,sensor.bev.newFxScale,sensor.bev.newFyScale)
        sensor.b_fish.H_cam2bev = calc_H(sensor.b_fish.ext_car2cam, sensor.b_fish.K, sensor.bev.virtual_K,sensor.bev.newFxScale,sensor.bev.newFyScale)
        sensor.l_fish.H_cam2bev = calc_H(sensor.l_fish.ext_car2cam, sensor.l_fish.K, sensor.bev.virtual_K,sensor.bev.newFxScale,sensor.bev.newFyScale)
        sensor.r_fish.H_cam2bev = calc_H(sensor.r_fish.ext_car2cam, sensor.r_fish.K, sensor.bev.virtual_K,sensor.bev.newFxScale,sensor.bev.newFyScale)
        sensor.front800.H_cam2bev = calc_H(sensor.front800.ext_car2cam, sensor.front800.K, sensor.bev.virtual_K) 
        
        sensor.f_fish.newFxScale = sensor.bev.newFxScale
        sensor.f_fish.newFyScale = sensor.bev.newFyScale
        sensor.b_fish.newFxScale = sensor.bev.newFxScale
        sensor.b_fish.newFyScale = sensor.bev.newFyScale
        sensor.l_fish.newFxScale = sensor.bev.newFxScale
        sensor.l_fish.newFyScale = sensor.bev.newFyScale
        sensor.r_fish.newFxScale = sensor.bev.newFxScale
        sensor.r_fish.newFyScale = sensor.bev.newFyScale
        
        
        
    return sensor

def undist_image(camera: CamParam, img):
    if camera.model == CamModel.FISHEYE:
        res = fish_undistort_img(img, camera.K, camera.D, camera.newFxScale, camera.newFyScale)
    elif camera.model == CamModel.PINHOLE:
        newK = copy.deepcopy(camera.K)
        newK[0,0] *= camera.newFxScale
        newK[1,1] *= camera.newFyScale
        res = cv2.undistort(img, camera.K, camera.D, None, newK)
    else:
        raise ValueError(f"{camera.model} not apply")
    
    return res

def undist_points(camera: CamParam, src_points, b_del_not_in_fov:bool=False):
    if camera.model == CamModel.FISHEYE:
        res = fish_undistort_points_FxyScale(src_points, camera.theta, camera.r_distorted, camera.K, camera.newFxScale, camera.newFyScale)
    elif camera.model == CamModel.PINHOLE:
        newK = copy.deepcopy(camera.K)
        newK[0,0] *= camera.newFxScale
        newK[1,1] *= camera.newFyScale
        res = cv2.undistortPoints(src_points, camera.K, camera.D, P=newK)
        res = res.reshape((-1,2))
    else:
        raise ValueError(f"{camera.model} not apply")
    
    # if b_del_not_in_fov:
    #     res = res[res[:, 0] >= 0]
    #     res = res[res[:, 1] >= 0]
    #     res = res[res[:, 0] < camera.img_w]
    #     res = res[res[:, 1] < camera.img_h]
    # else:
    #     res[res[:, 0] < 0,0] = 0
    #     res[res[:, 1] < 0,1] = 0
    #     res[res[:, 0] >= camera.img_w,0]=camera.img_w-1
    #     res[res[:, 1] >= camera.img_h,1]=camera.img_h-1
    
    return res


# 
sensor = Calib_Parse()

#
def get_offx_offy_camera_id(cam_id: str):
    if cam_id == "cam2":
        offx = 0
        offy = 10000
    elif cam_id == "cam9":
        offx = 0
        offy = 10000
    elif cam_id == "cam10":
        offx = 2000
        offy = 0
    elif cam_id == "cam11":
        offx = -2000
        offy = 0
    elif cam_id == "cam12":
        offx = 0
        offy = -10000
    else:
        raise ValueError(f"{cam_id} not exitst")
    
    return offx,offy


def dist_image_to_bev_camera_id_fblr_dis(
    cam_id: str, 
    img, img_h:int, 
    img_w:int, 
    bev_h:int, 
    bev_w:int,
    fdis: int, bdis:int, ldis:int, rdis:int
    ):
    if cam_id == "cam2":
        camera = sensor.front800
    elif cam_id == "cam9":
        camera = sensor.f_fish
    elif cam_id == "cam10":
        camera = sensor.l_fish
    elif cam_id == "cam11":
        camera = sensor.r_fish
    elif cam_id == "cam12":
        camera = sensor.b_fish
    else:
        raise ValueError(f"{cam_id} not exitst")
    #
    img_crop = img[:img_h,:img_w]
    img_resize = cv2.resize(img_crop, (camera.img_w,camera.img_h))
    img_undist = undist_image(camera, img_resize)
    #
    bev_ins = copy.deepcopy(sensor.bev)
    bev_ins.f_dis = fdis
    bev_ins.b_dis = bdis
    bev_ins.l_dis = ldis
    bev_ins.r_dis = rdis
    bev_ins.bev_width = int((bev_ins.l_dis+bev_ins.r_dis)/bev_ins.resolution)
    bev_ins.bev_height= int((bev_ins.f_dis+bev_ins.b_dis)/bev_ins.resolution)        
    bev_ins.virtual_K = calc_bev_K(bev_ins)
    new_bev_k = calc_bev_K(bev_ins)
    new_H_cam2bev = calc_H(camera.ext_car2cam,camera.K,new_bev_k,camera.newFxScale,camera.newFyScale)
    
    bev_img = cv2.warpPerspective(img_undist, new_H_cam2bev, (bev_ins.bev_width, bev_ins.bev_height))
    bev_img = cv2.resize(bev_img,(bev_w, bev_h))
    
    return bev_img

def undist_points_to_ground_points(undistPixels, K, Rt):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    undistPixels_normalized = np.zeros_like(undistPixels)
    undistPixels_normalized[:, 0] = (undistPixels[:, 0] - cx) / fx
    undistPixels_normalized[:, 1] = (undistPixels[:, 1] - cy) / fy
    undistPixels_normalized_homogeneous = np.hstack((undistPixels_normalized, np.ones((undistPixels.shape[0], 1))))

    T = np.array(Rt, dtype=np.float32)
    R = T[:3, :3]
    t = T[:3, 3]

    cam_center = np.array([0, 0, 0], dtype=np.float32)
    cam_center_ground = np.dot(np.linalg.inv(R), cam_center - t)

    # print(cam_center_ground)

    corners_ground = np.dot(np.linalg.inv(R), undistPixels_normalized_homogeneous.T - t.reshape((3, 1))).T

    # print(corners_ground)

    direct_v = corners_ground.T - cam_center_ground.reshape((3, 1))
    length = np.linalg.norm(direct_v, axis=0)
    normalized_v = direct_v / length
    scale = -1*cam_center_ground[2] / normalized_v[2, :]
    scale = scale.reshape((1, -1))

    ground_points = cam_center_ground.reshape((3, 1)) + scale*normalized_v
    return ground_points.T
    
def dist_points_to_ground_points(cam_id: str, src_points):
    # src_points: N*2
    
    if cam_id == "cam2":
        camera = sensor.front800
    elif cam_id == "cam9":
        camera = sensor.f_fish
    elif cam_id == "cam10":
        camera = sensor.l_fish
    elif cam_id == "cam11":
        camera = sensor.r_fish
    elif cam_id == "cam12":
        camera = sensor.b_fish
    else:
        raise ValueError(f"{cam_id} not exitst")
    
    points_undist = undist_points(camera, src_points)
    newCamMatrix = np.copy(camera.K)
    newCamMatrix[0, 0] *= camera.newFxScale
    newCamMatrix[1, 1] *= camera.newFyScale
    
    ground_points = undist_points_to_ground_points(points_undist, newCamMatrix, camera.ext_car2cam)
    return ground_points
    
def dist_points_to_bev_camera_id_fblr_dis(
    cam_id: str, 
    src_points, #N*2 
    img_h:int, 
    img_w:int, bev_h:int, bev_w:int,
    fdis: int, bdis:int, ldis:int, rdis:int,
    b_del_not_in_fov:bool=False
    ):
    if cam_id == "cam2":
        camera = sensor.front800
    elif cam_id == "cam9":
        camera = sensor.f_fish
    elif cam_id == "cam10":
        camera = sensor.l_fish
    elif cam_id == "cam11":
        camera = sensor.r_fish
    elif cam_id == "cam12":
        camera = sensor.b_fish
    else:
        raise ValueError(f"{cam_id} not exitst")
    
    #
    scale_x = camera.img_w/img_w
    scale_y = camera.img_h/img_h
    
    points_resize = copy.deepcopy(src_points)
    points_resize[:,0] *= scale_x
    points_resize[:,1] *= scale_y

    #
    points_undist = undist_points(camera, points_resize, b_del_not_in_fov)
    
    #
    bev_ins = copy.deepcopy(sensor.bev)
    bev_ins.f_dis = fdis
    bev_ins.b_dis = bdis
    bev_ins.l_dis = ldis
    bev_ins.r_dis = rdis
    bev_ins.bev_width = int((bev_ins.l_dis+bev_ins.r_dis)/bev_ins.resolution)
    bev_ins.bev_height= int((bev_ins.f_dis+bev_ins.b_dis)/bev_ins.resolution)        
    bev_ins.virtual_K = calc_bev_K(bev_ins)
    new_bev_k = calc_bev_K(bev_ins)
    new_H_cam2bev = calc_H(camera.ext_car2cam,camera.K,new_bev_k,camera.newFxScale,camera.newFyScale)
    bev_points = undisPoints2bev(points_undist, new_H_cam2bev)
    
    #
    scale_x = bev_w/bev_ins.bev_width
    scale_y = bev_h/bev_ins.bev_height
    
    bev_points[:,0] *= scale_x
    bev_points[:,1] *= scale_y
    
    #
    if camera.bev_filter_type == BevFilterType.FilterF:
        bev_points =  bev_points[bev_points[:, 1] >= bev_h*fdis/(fdis+bdis)]
    elif camera.bev_filter_type == BevFilterType.FilterB:
        bev_points =  bev_points[bev_points[:, 1] <= bev_h*fdis/(fdis+bdis)]
    elif camera.bev_filter_type == BevFilterType.FilterL:
        bev_points =  bev_points[bev_points[:, 0] >= bev_w*ldis/(ldis+rdis)]
    elif camera.bev_filter_type == BevFilterType.FilterR:
        bev_points =  bev_points[bev_points[:, 0] <= bev_w*ldis/(ldis+rdis)]
        
    if b_del_not_in_fov:
        bev_points =  bev_points[bev_points[:, 0] >= 0]
        bev_points =  bev_points[bev_points[:, 1] >= 0]
        bev_points =  bev_points[bev_points[:, 0] <= bev_w-1]
        bev_points =  bev_points[bev_points[:, 1] <= bev_h-1]
    else:
        bev_points[bev_points[:, 0] < 0,0] = 0
        bev_points[bev_points[:, 1] < 0,1] = 0
        bev_points[bev_points[:, 0] >= bev_w-1,0]=bev_w-1
        bev_points[bev_points[:, 1] >= bev_h-1,1]=bev_h-1
    
    return bev_points
   

def dist_image_to_bev_camera_id(cam_id: str, img, img_h:int, img_w:int, bev_h:int, bev_w:int):
    if cam_id == "cam2":
        camera = sensor.front800
    elif cam_id == "cam9":
        camera = sensor.f_fish
    elif cam_id == "cam10":
        camera = sensor.l_fish
    elif cam_id == "cam11":
        camera = sensor.r_fish
    elif cam_id == "cam12":
        camera = sensor.b_fish
    else:
        raise ValueError(f"{cam_id} not exitst")
    
    #
    offx,offy = get_offx_offy_camera_id(cam_id)
    
    #
    img_crop = img[:img_h,:img_w]
    img_resize = cv2.resize(img_crop, (camera.img_w,camera.img_h))
    img_undist = undist_image(camera, img_resize)
    #
    bev_ins = copy.deepcopy(sensor.bev)
    bev_ins.offx = offx
    bev_ins.offy = offy
    new_bev_k = calc_bev_K(bev_ins)
    new_H_cam2bev = calc_H(camera.ext_car2cam,camera.K,new_bev_k,camera.newFxScale,camera.newFyScale)
    
    bev_img = cv2.warpPerspective(img_undist, new_H_cam2bev, (bev_ins.bev_width+abs(offx), bev_ins.bev_height+abs(offy)))
    bev_img = cv2.resize(bev_img,(int(bev_w+abs(offx)*bev_w/bev_ins.bev_width), int(bev_h+abs(offy)*bev_h/bev_ins.bev_height)))
    
    return bev_img

def dist_points_to_bev_camera_id(cam_id: str, src_points, img_h:int, img_w:int, bev_h:int, bev_w:int):
    if cam_id == "cam2":
        camera = sensor.front800
    elif cam_id == "cam9":
        camera = sensor.f_fish
    elif cam_id == "cam10":
        camera = sensor.l_fish
    elif cam_id == "cam11":
        camera = sensor.r_fish
    elif cam_id == "cam12":
        camera = sensor.b_fish
    else:
        raise ValueError(f"{cam_id} not exitst")
    
    #
    offx,offy = get_offx_offy_camera_id(cam_id)
    
    #
    scale_x = camera.img_w/img_w
    scale_y = camera.img_h/img_h
    
    points_resize = copy.deepcopy(src_points)
    points_resize[:,0] *= scale_x
    points_resize[:,1] *= scale_y

    #
    points_undist = undist_points(camera, points_resize)
    
    #
    bev_ins = copy.deepcopy(sensor.bev)
    bev_ins.offx = offx
    bev_ins.offy = offy
    new_bev_k = calc_bev_K(bev_ins)
    new_H_cam2bev = calc_H(camera.ext_car2cam,camera.K,new_bev_k,camera.newFxScale,camera.newFyScale)
    bev_points = undisPoints2bev(points_undist, new_H_cam2bev)
    
    #
    scale_x = bev_w/sensor.bev.bev_width
    scale_y = bev_h/sensor.bev.bev_height
    
    bev_points[:,0] *= scale_x
    bev_points[:,1] *= scale_y
    
    #
    if camera.bev_filter_type == BevFilterType.FilterF:
        bev_points =  bev_points[bev_points[:, 1] >= bev_h*0.5-1]
    elif camera.bev_filter_type == BevFilterType.FilterB:
        bev_points =  bev_points[bev_points[:, 1] <= (sensor.bev.bev_height+abs(offy))*scale_y-bev_h*0.5+1]
    elif camera.bev_filter_type == BevFilterType.FilterL:
        bev_points =  bev_points[bev_points[:, 0] >= bev_w*0.5-1]
    elif camera.bev_filter_type == BevFilterType.FilterR:
        bev_points =  bev_points[bev_points[:, 0] <= (sensor.bev.bev_width+abs(offx))*scale_x-bev_w*0.5+1]
        
    bev_points[bev_points[:, 0] < 0,0] = 0
    bev_points[bev_points[:, 1] < 0,1] = 0
    bev_points[bev_points[:, 0] >= (sensor.bev.bev_width+abs(offx))*scale_x,0]=(sensor.bev.bev_width+abs(offx))*scale_x-1
    bev_points[bev_points[:, 1] >= (sensor.bev.bev_height+abs(offy))*scale_y,1]=(sensor.bev.bev_height+abs(offy))*scale_y-1
    
    return bev_points

if __name__ == '__main__':
    sensor = Calib_Parse()
    
    # path = "/perception/users/fanjin/1025/ip16/5_camera_2023-10-25-14-50-00_2/sensor/front_fish/1698216601.586254000.jpg"
    # img = cv2.imread(path)
    # img_undist = fish_undistort_img(img, sensor.f_fish.K, sensor.f_fish.D, sensor.bev.newFxScale, sensor.bev.newFyScale)
    # bev_img = cv2.warpPerspective(img_undist, sensor.f_fish.H_cam2bev, (sensor.bev.bev_width, sensor.bev.bev_height))
    
    # path = "/perception/users/fanjin/1025/ip16/5_camera_2023-10-25-14-50-00_2/sensor/camera_800/1698216601.484543000.jpg"
    # img = cv2.imread(path)
    # img_undist = cv2.undistort(img, sensor.front800.K, sensor.front800.D)
    # bev_img = cv2.warpPerspective(img_undist, sensor.front800.H_cam2bev, (sensor.bev.bev_width, sensor.bev.bev_height))
    
    
    print(sensor.front800.ext_car2cam)
    print(sensor.front800.K)
    print(sensor.front800.D)
    print(sensor.front800.H_cam2bev)
    print(sensor.front800.newFxScale)
    print(sensor.front800.newFyScale)

    path = "/home/fanjin/test.jpg"
    img = cv2.imread(path)
    img_undist = cv2.undistort(img, sensor.front800.K, sensor.front800.D)
    bev_img = cv2.warpPerspective(img_undist, sensor.front800.H_cam2bev, (sensor.bev.bev_width, sensor.bev.bev_height))
    
    
    bev_img = dist_image_to_bev_camera_id("cam2", img, 3840, 2160 ,640, 512)
    
    print("finish")