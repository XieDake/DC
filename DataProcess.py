# -*- coding: utf-8 -*-
#import some packages!
import numpy as np
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
    def __init__(self,filename_train_positive='TrainingSet_positive_1.txt',
                 filename_train_negtive='TrainingSet_negtive_1.txt',
                 filename_test='TestingSet_1.txt',filename_train='TrainingSet_1.txt'):
        self.fileName_train_positive=filename_train_positive
        self.fileName_train_negtive=filename_train_negtive
        self.fileName_test=filename_test
        self.fileName_train=filename_train
        #numpy array!
        self.trainingSet_positive = np.loadtxt(self.fileName_train_positive, delimiter=';', dtype=float)
        self.trainingSet_negtive = np.loadtxt(self.fileName_train_negtive, delimiter=';', dtype=float)
        self.testingSet = np.loadtxt(self.fileName_test, delimiter=';', dtype=float)
        self.trainingSet = np.loadtxt(self.fileName_train, delimiter=';', dtype=float)#某些模型不必划分正负样本！
        #initialize end!
    #positive/negtive sample!
    #由于正（0）：负（1）=48413：7183，
    #所以负样本全部使用，控制正样本的比例即可！
    #ratio（正：负）：{3;6.7}
    #同时划分数据集：train（7）：validation（3）
    #38917:16679
    #这里就先不考虑ratio的事情了！
    #默认6.7
    def posAndNegtive(self):
        '''
        话说这个方法已经没用咯！
        retio moren6.7：1，因为在这里改变ratio没意义！因为不知道test中ratio是多少！只能在线上测试的时候挨个试了！
        到时候此程序得重写！
        :return:
        '''
        ftr_pos=self.trainingSet_positive
        ftr_neg=self.trainingSet_negtive
        m1,n1=np.shape(ftr_pos)
        m2,n2=np.shape(ftr_neg)
        traingSet_final=np.zeros([38917,n1])
        validationSet_final=np.zeros([16679,n2])
        #training:38917
        traingSet_final[:33889,:]=ftr_pos[:33889,:]
        traingSet_final[33889:,:]=ftr_neg[:5028,:]
        #validation:16679
        validationSet_final[:14524,:]=ftr_pos[33889:,:]
        validationSet_final[14524:,:]=ftr_neg[5028:,:]
        return traingSet_final,validationSet_final
    #划分数据集！
    #正负样本：ratio，负样本全用，，控制正样本的数目！
    #负样本：6605个:train:5284 validation:1321
    #ratio=正：负
    def posAndNegtive1(self,ratio):
        ftr_pos=self.trainingSet_positive
        ftr_neg=self.trainingSet_negtive
        m,n=ftr_neg.shape
        #先把正样本分两份8：2;
        #train_pos_all=np.zeros([37255,n])
        train_pos_all=ftr_pos[:37255,:]#37255
        #validatiion_pos_all=np.zeros([9314,n])
        validatiion_pos_all=ftr_pos[37255:,:]#9314
        m1_pos=int(5284*ratio)#train中正样本数目！
        m1=m1_pos+5284#只改变train中正样本的比例！负样本5284不变
        trainingSet_final=np.zeros([m1,n])
        trainingSet_final[:m1_pos,:]=train_pos_all[:m1_pos,:]#正入m1_pos个！
        trainingSet_final[m1_pos:,:]=ftr_neg[:5284,:]#负入5284个！
        m2_pos=int(1321*ratio)
        m2=m2_pos+1321#只改变validation中正样本的比例！负样本1437不变！
        validationSet_final=np.zeros([m2,n])
        validationSet_final[:m2_pos, :] = validatiion_pos_all[:m2_pos, :]  # 正入m2_pos个！
        validationSet_final[m2_pos:, :] = ftr_neg[5284:, :]  # 负入1321个！
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
        trainingSet_final_use[:m1_pos,:]=ftr_pos[:m1_pos,:]#证入m1_pos个！
        trainingSet_final_use[m1_pos:,:]=ftr_neg#负全入：6605个！
        return trainingSet_final_use