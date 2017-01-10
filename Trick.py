# -*- coding: utf-8 -*-
'''
处理哪些比较异常的，基本没有预测出来的结果！
均值不解释！
'''
#import
import numpy as np
#my trick!
def myTrick():
    '''
    文件必须排好序,一样的order！！
    :return:
    '''
    fileName_new = 'result_201716.txt'
    fileName_old = 'result_201715.txt'
    f_new = np.loadtxt(fileName_new, delimiter='\t', dtype=float)
    f_old = np.loadtxt(fileName_old, delimiter='\t', dtype=float)
    m,n=f_new.shape
    fresult=np.zeros([m,n])
    for i in range(m):
        #(13899,2),index:0_13898
        if (f_new[i][1]!=0.86664327299999999):
            fresult[i]=f_new[i]
        else:
            fresult[i][0]=f_new[i][0]
            fresult[i][1]=(f_new[i][1]+f_old[i][1])/2
    np.savetxt('result_201717_trick.csv', fresult, delimiter=',')


