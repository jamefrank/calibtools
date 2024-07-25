'''
Author: fanjin 
Date: 2024-07-17 19:42:45
LastEditors: fanjin 
LastEditTime: 2024-07-18 10:04:44
FilePath: /calibtools/calibtools/opt_utils/opt.py
Description: 常用的优化工具

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

import numpy as np

def opt_R_by_direction(directs1: np.ndarray, directs2: np.ndarray):
    '''
    根据同名向量计算旋转矩阵:R_12
    directs1: N*3
    directs2: N*3
    '''
    
    # normalize
    directs1 = directs1 / np.linalg.norm(directs1, axis=1, keepdims=True)
    directs2 = directs2 / np.linalg.norm(directs2, axis=1, keepdims=True)
    
    # cov
    cov = directs2.T @ directs1
    u, s, vh = np.linalg.svd(cov)
    det = np.linalg.det(vh @ u.T)
    I = np.identity(3)
    if det < 0:
        I[2, 2] = -1
    R = vh.T @ I @ u.T
    
    
    
    pass


def main():
    pass
 
 
if __name__ == '__main__':
    main()
    pass
