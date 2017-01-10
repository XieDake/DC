# -*- coding: utf-8 -*-
#Kolmogrov-Smirnov(KS) Score
#先排好序噻！按最后一列从小到大整体排序！
#import
import numpy as np
#
def sort_up(f):
        '''
        :param f:
        :return: l_in是set类型！
        '''
        #先把f最后一列排序的index得到!
        index=np.argsort(f[:,-1])
        #再按index的索引顺序！从小到大重新排序！
        m,n=np.shape(f)
        f_new=np.zeros([m,n])
        count=0
        for i in index:
            f_new[count,:]=f[i,:]
            count=count+1
        #返回最后一列中不重复的元素！
        l_all=[]
        for i in f[:,-1]:
            l_all.append(i)
        #list 去重！返回set类型l_in
        l_in=set(l_all)
        return f_new,l_in
#计算KS评分！
def ks_score(f):
        f_new,l_in=sort_up(f)
        #按照计算规则开始计算！
        #f:每一列的含义：[user_id,lable，prob]
        count_0=0
        count_1=0
        max_score=0
        for i in l_in:
            for j in f_new:
                if(j[2]==i):
                    if(j[1]==0):
                        count_0+=1
                    else:
                        count_1+=1
            #对每一个i都遍历一遍f_new
            #内循环遍历完事后！计算当前的差!
            temp=abs(count_0-count_1)
            if(temp>max_score):
                max_score=temp
        #外循环结束后返回score！
        return max_score
#ks_score 测试：
def ks_score_test():
        f_test=np.array([[0,1,0.8],
                         [1,0,0.2],
                         [2,1,0.7],
                         [3,0,0.3],
                         [4,1,0.5],
                         [5,0,0.5],
                         [6,1,0.5],
                         [7,0,0.7],
                         [8,1,0.7],
                         [9,0,0.2]])
        Ks_score=ks_score(f_test)
        print Ks_score