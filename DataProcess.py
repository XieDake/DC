# -*- coding: utf-8 -*-
#import some packages!
import numpy as np
import pandas as pd
#class...go!
class MyDataProcess:
    fileName_train=''
    fileName_train_positive=''
    fileName_train_negtive=''
    fileName_test=''
    trainingSet=''
    trainingSet_positive=''
    trainingSet_negtive=''
    testingSet=''
    def __init__(self,filename_train_positive='TrainingSet_positive_2017112.txt',
                 filename_train_negtive='TrainingSet_negtive_2017112.txt',
                 filename_test='TestingSet_2017112.txt',filename_train='TrainingSet_2017112.txt'):
        self.fileName_train_positive=filename_train_positive
        self.fileName_train_negtive=filename_train_negtive
        self.fileName_test=filename_test
        self.fileName_train=filename_train
        #numpy array!
        #self.trainingSet_positive = np.loadtxt(self.fileName_train_positive, delimiter=';', dtype=float)
        #self.trainingSet_negtive = np.loadtxt(self.fileName_train_negtive, delimiter=';', dtype=float)
        #self.testingSet = np.loadtxt(self.fileName_test, delimiter=';', dtype=float)
        #self.trainingSet = np.loadtxt(self.fileName_train, delimiter=';', dtype=float)#某些模型不必划分正负样本！
        #panda dataFrame类型！
        self.trainingSet_positive=pd.read_csv(self.fileName_train_positive,header=None,sep=';')
        self.trainingSet_negtive=pd.read_csv(self.fileName_train_negtive,header=None,sep=';')
        self.testingSet=pd.read_csv(self.fileName_test,header=None,sep=';').values
        self.trainingSet=pd.read_csv(self.fileName_train,header=None,sep=';').values
        #initialize end!
    #positive/negtive sample!
    #由于正（0）：负（1）=48413：7183，
    #所以负样本全部使用，控制正样本的比例即可！
    #ratio（正：负）：{3;6.7}
    #同时划分数据集：train（7）：validation（3）
    #38917:16679
    #这里就先不考虑ratio的事情了！
    #默认6.7
    #划分数据集！
    #正负样本：ratio，负样本全用，，控制正样本的数目！
    #负样本：6605个:train:5284 validation:1321
    #ratio=正：负
    def posAndNegtive1(self,ratio_tr,ratio_val):
        ftr_pos=self.trainingSet_positive
        ftr_neg=self.trainingSet_negtive
        m,n=ftr_neg.shape
        #先把正样本分两份8：2;
        #train_pos_all=np.zeros([37255,n])
        train_pos_all=ftr_pos.iloc[:37255,:]#37255
        #validatiion_pos_all=np.zeros([9314,n])
        validatiion_pos_all=ftr_pos.iloc[37255:,:]#9314
        #train
        m1_pos=int(5284*ratio_tr)#train中正样本数目！
        m1=m1_pos+5284#只改变train中正样本的比例！负样本5284不变
        trainingSet_final=np.zeros([m1,n])#numpy形式！
        # 正入m1_pos个！
        train_pos_part=train_pos_all.sample(m1_pos) #随机取样m1_pos个！！不放回采样！当然不能重复咯！
        trainingSet_final[:m1_pos,:]=train_pos_part.values #正入m1_pos个！numpy形式！
        # 负入5284个！
        train_neg_part=ftr_neg.iloc[:5284,:]
        trainingSet_final[m1_pos:,:]=train_neg_part.values #负入5284个！numpy形式！
        #validation!
        m2_pos=int(1321*ratio_val)
        m2=m2_pos+1321#只改变validation中正样本的比例！负样本1437不变！
        validationSet_final=np.zeros([m2,n])
        # 正入m2_pos个！
        validatiion_pos_part=validatiion_pos_all.sample(m2_pos)#随机采样m2_pos个！无放回采样!
        validationSet_final[:m2_pos, :] = validatiion_pos_part.values  # 正入m2_pos个！numpy形式！
        # 负入1321个！
        validatiion_neg_part=ftr_neg.iloc[5284:, :]
        validationSet_final[m2_pos:, :] = validatiion_neg_part.values  # 负入1321个！numpy形式！
        return trainingSet_final,validationSet_final
    def trainAllUsing(self,ratio):
        '''
        为了充分使用样本！最后选好参数后，再使用Alldata训练一下：
        负样本1：6605全部使用！
        正样本0：6605*ratio!
        :return:
        '''
        ftr_pos=self.trainingSet_positive
        ftr_neg=self.trainingSet_negtive
        m,n=ftr_pos.shape
        m1_neg=6605#负样本全用！
        m1_pos=int(m1_neg*ratio)
        m1=m1_neg+m1_pos
        trainingSet_final_use=np.zeros([m1,n])
        trainingSet_final_use[:m1_pos,:]=(ftr_pos.iloc[:m1_pos,:]).values#证入m1_pos个！
        trainingSet_final_use[m1_pos:,:]=ftr_neg.values#负全入：6605个！
        return trainingSet_final_use