'''
Author: fanjin jinfan.@novauto.com.cn
Date: 2024-07-13 22:48:32
LastEditors: fanjin jinfan.@novauto.com.cn
LastEditTime: 2024-07-13 22:48:35
FilePath: /calibtools/calibtools/camera/common.py
Description: 

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

from enum import Enum

class CamModel(Enum):
    PINHOLE = 'pinhole'
    FISH = 'fish'
    