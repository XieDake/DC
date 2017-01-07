# -*- coding: utf-8 -*-
'''
处理哪些比较异常的，基本没有预测出来的结果！
均值不解释！
'''
#import
import numpy as np
#my trick!
def myTrick():
    fileName_new = 'result_201716.txt'
    fileName_old = 'result_201715.txt'
    f_new = np.loadtxt(fileName_new, delimiter='\t', dtype=float)
    f_old = np.loadtxt(fileName_old, delimiter='\t', dtype=float)
    m,n=f_new.shape
