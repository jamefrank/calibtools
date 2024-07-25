'''
Author: fanjin 
Date: 2024-07-19 12:42:29
LastEditors: fanjin 
LastEditTime: 2024-07-19 12:50:30
FilePath: /calibtools/calibtools/calib_res_utils/parse.py
Description: 解析标定结果文件

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''


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
class XingYueSensorConfig:
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




def XingYue_Calib_Parse():
    """解析传感器的内外参
    """
    sensor = XingYueSensorConfig()
    calib_file = Path(__file__).parent/"calib.yaml"
    with open(calib_file, 'r') as file:
        config = yaml.safe_load(file)
        # 解析相机参数
        sensor.f_fish.K = np.array(config['FrontFish']['K'], dtype=np.float64)
        sensor.f_fish.D = np.array(config['FrontFish']['D'], dtype=np.float64)
        sensor.f_fish.theta, sensor.f_fish.r_distorted = fish_distorted_table(sensor.f_fish.D)
        sensor.f_fish.ext_car2cam = np.array(config['TGroundFrontFish'], dtype=np.float64)
        sensor.f_fish.img_h = 1080
        sensor.f_fish.img_w = 1920
        sensor.f_fish.model = CamModel.FISHEYE
        sensor.f_fish.bev_filter_type = BevFilterType.FilterB
        
        sensor.b_fish.K = np.array(config['BackFish']['K'], dtype=np.float64)
        sensor.b_fish.D = np.array(config['BackFish']['D'], dtype=np.float64)
        sensor.b_fish.theta, sensor.b_fish.r_distorted = fish_distorted_table(sensor.b_fish.D)
        sensor.b_fish.ext_car2cam = np.array(config['TGroundBackFish'], dtype=np.float64)
        sensor.b_fish.img_h = 1080
        sensor.b_fish.img_w = 1920
        sensor.b_fish.model = CamModel.FISHEYE
        sensor.b_fish.bev_filter_type = BevFilterType.FilterF

        sensor.l_fish.K = np.array(config['LeftFish']['K'], dtype=np.float64)
        sensor.l_fish.D = np.array(config['LeftFish']['D'], dtype=np.float64)
        sensor.l_fish.theta, sensor.l_fish.r_distorted = fish_distorted_table(sensor.l_fish.D)
        sensor.l_fish.ext_car2cam = np.array(config['TGroundLeftFish'], dtype=np.float64)
        sensor.l_fish.img_h = 1080
        sensor.l_fish.img_w = 1920
        sensor.l_fish.model = CamModel.FISHEYE
        sensor.l_fish.bev_filter_type = BevFilterType.FilterR

        sensor.r_fish.K = np.array(config['RightFish']['K'], dtype=np.float64)
        sensor.r_fish.D = np.array(config['RightFish']['D'], dtype=np.float64)
        sensor.r_fish.theta, sensor.r_fish.r_distorted = fish_distorted_table(sensor.r_fish.D)
        sensor.r_fish.ext_car2cam = np.array(config['TGroundRightFish'], dtype=np.float64)
        sensor.r_fish.img_h = 1080
        sensor.r_fish.img_w = 1920
        sensor.r_fish.model = CamModel.FISHEYE
        sensor.r_fish.bev_filter_type = BevFilterType.FilterL

        sensor.front800.K = np.array(config['Front800']['K'], dtype=np.float64)
        sensor.front800.D = np.array(config['Front800']['D'], dtype=np.float64)
        sensor.front800.ext_car2cam = np.array(config['TGround800'], dtype=np.float64)
        sensor.front800.img_h = 2160
        sensor.front800.img_w = 3840
        
        sensor.front200.K = np.array(config['Front200']['K'], dtype=np.float64)
        sensor.front200.D = np.array(config['Front200']['D'], dtype=np.float64)
        sensor.front200.img_h = 1080
        sensor.front200.img_w = 1920  #TODO to valid?
        
        sensor.back200.K = np.array(config['Back200']['K'], dtype=np.float64)
        sensor.back200.D = np.array(config['Back200']['D'], dtype=np.float64)
        sensor.back200.img_h = 1080
        sensor.back200.img_w = 1920  #TODO to valid?
        sensor.back200.ext_car2cam = np.array(config['TGroundBack200'], dtype=np.float64)

        # 外参
        sensor.at128To800 = np.array(config['T_800_at128'],dtype=np.float64)
        sensor.backRslidarToback200 = np.array(config['T_back200_rslidar'],dtype=np.float64)

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


# G3
@dataclass
class G3SensorConfig:
    f_fish: CamParam = CamParam()
    b_fish: CamParam = CamParam()
    l_fish: CamParam = CamParam()
    r_fish: CamParam = CamParam()
    front800: CamParam = CamParam()
    
    scan_lb: CamParam = CamParam()
    scan_lf: CamParam = CamParam()
    scan_rb: CamParam = CamParam()
    scan_rf: CamParam = CamParam()
    scan_f: CamParam = CamParam()
    scan_b: CamParam = CamParam()
    
    bev: BevInsParam = BevInsParam()
    tanwayTo800: np.ndarray=None
    carToqbxg: np.ndarray=None
    tanwayToqbxg: np.ndarray=None

def G3_Calib_Parse():
    """解析传感器的内外参
    """
    sensor = G3SensorConfig()
    calib_file = Path(__file__).parent/"calib-g3.yaml"
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
        
        sensor.scan_lb.K = np.array(config['SideLB']['K'], dtype=np.float64)
        sensor.scan_lb.D = np.array(config['SideLB']['D'], dtype=np.float64)
        # sensor.scan_lb.ext_car2cam = np.array(config['TGround800'], dtype=np.float64)  #TODO
        sensor.scan_lb.img_h = 1080
        sensor.scan_lb.img_w = 1920
        
        sensor.scan_lf.K = np.array(config['SideLF']['K'], dtype=np.float64)
        sensor.scan_lf.D = np.array(config['SideLF']['D'], dtype=np.float64)
        # sensor.scan_lf.ext_car2cam = np.array(config['TGround800'], dtype=np.float64)  #TODO
        sensor.scan_lf.img_h = 1080
        sensor.scan_lf.img_w = 1920
        
        sensor.scan_rb.K = np.array(config['SideRB']['K'], dtype=np.float64)
        sensor.scan_rb.D = np.array(config['SideRB']['D'], dtype=np.float64)
        # sensor.scan_rb.ext_car2cam = np.array(config['TGround800'], dtype=np.float64)  #TODO
        sensor.scan_rb.img_h = 1080
        sensor.scan_rb.img_w = 1920
        
        sensor.scan_rf.K = np.array(config['SideRF']['K'], dtype=np.float64)
        sensor.scan_rf.D = np.array(config['SideRF']['D'], dtype=np.float64)
        # sensor.scan_rf.ext_car2cam = np.array(config['TGround800'], dtype=np.float64)  #TODO
        sensor.scan_rf.img_h = 1080
        sensor.scan_rf.img_w = 1920
        
        sensor.scan_b.K = np.array(config['SideB']['K'], dtype=np.float64)
        sensor.scan_b.D = np.array(config['SideB']['D'], dtype=np.float64)
        # sensor.scan_b.ext_car2cam = np.array(config['TGround800'], dtype=np.float64)  #TODO
        sensor.scan_b.img_h = 1080
        sensor.scan_b.img_w = 1920
        
        sensor.scan_f.K = np.array(config['SideF']['K'], dtype=np.float64)
        sensor.scan_f.D = np.array(config['SideF']['D'], dtype=np.float64)
        # sensor.scan_f.ext_car2cam = np.array(config['TGround800'], dtype=np.float64)  #TODO
        sensor.scan_f.img_h = 1080
        sensor.scan_f.img_w = 1920
        
        

        # 外参
        sensor.tanwayTo800 = np.array(config['T_800_tanway'],dtype=np.float64)
        sensor.carToqbxg = np.array(config['car_to_qbxg'],dtype=np.float64)
        sensor.tanwayToqbxg = sensor.carToqbxg @ np.linalg.inv(sensor.front800.ext_car2cam) @ sensor.tanwayTo800

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
  

if __name__ == '__main__':    
    sensor = G3_Calib_Parse()
    print(sensor.tanwayToqbxg)