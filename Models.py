# -*- coding: utf-8 -*-
#import!
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#from sklearn import cross_validation, metrics
#from sklearn.model_selection import GridSearchCV
#ending!
import DataProcess as Q1
#models
class Models:
    traingSet=''
    traingSet_all=''
    validationSet=''
    test=''
    Dp=''
    def __init__(self,ratio_tr,ratio_val):
        self.Dp=Q1.MyDataProcess()
        self.traingSet,self.validationSet=self.Dp.posAndNegtive1(ratio_tr,ratio_val)
        self.test=self.Dp.testingSet
        self.traingSet_all=self.Dp.trainAllUsing(ratio_val)
        #self.traingSet1,self.validationSet1=self.Dp.devide()
        #self.traingSet_final,self.validationSet_final=self.Dp.posAndNegtive1(ratio)
        #ending!
    #逻辑回归选择合适的C；
    def logRCV(self):
        ftr=self.traingSet
        print ftr.shape
        fval=self.validationSet
        print fval.shape
        C = [0.01,0.03, 0.1, 0.3, 1, 3, 10, 30]
        train_score=[]
        validation_score=[]
        for c in C:
            LR=LogisticRegression(C=c,penalty='l2')
            print c
            #train!
            LR.fit(ftr[:,1:-1],ftr[:,-1])
            print LR.score(ftr[:,1:-1],ftr[:,-1])
            train_score.append(LR.score(ftr[:,1:-1],ftr[:,-1]))
            #validation!
            validation_score.append(LR.score(fval[:,1:-1],fval[:,-1]))
            print LR.score(fval[:,1:-1],fval[:,-1])
        #plot
        print train_score
        print validation_score
        plt.plot(C,train_score,c='r')
        plt.plot(C,validation_score,c='b')
        plt.show()

    def logRCV1(self,c):
        ftr=self.traingSet
        fval=self.validationSet
        ftest = self.test
        print ftr.shape
        print fval.shape
        LR = LogisticRegression(C=c, penalty='l2')
        LR.fit(ftr[:, 1:-1], ftr[:, -1])
        y2=ftr[:,-1]
        ytr = LR.predict(ftr[:, 1:-1])
        yval=LR.predict(fval[:, 1:-1])
        ytest=LR.predict(ftest[:,1:])
        train_evalute=classification_report(ftr[:,-1], ytr)
        validation_evalute = classification_report(fval[:, -1], yval)
        print LR.classes_
        print train_evalute
        print validation_evalute
        #print LR.predict_proba(ftest[:,1:])
        #return ftest
    #逻辑回归输出！
    def logRCVOutPut(self,c):
        ftr=self.traingSet
        fval=self.validationSet
        ftest = self.test
        m,n=ftest.shape
        LR = LogisticRegression(C=c, penalty='l2')
        LR.fit(ftr[:, 1:-1], ftr[:, -1])
        ytr = LR.predict(ftr[:, 1:-1])
        yval=LR.predict(fval[:, 1:-1])
        ytest=LR.predict(ftest[:,1:])
        test_prob=LR.predict_proba(ftest[:,1:])
        test_output=np.zeros([m,2])
        test_output[:,0]=ftest[:,0]
        test_output[:,1]=test_prob[:,1]
        np.savetxt('result_201715.csv',test_output,delimiter=',')
        #return ftest
    #ef logRegression(self,C):
    #神经网络！双隐层！
    def NNS(self,m1,m2):
        ftr=self.traingSet
        fval=self.validationSet
        ftest=self.test
        classifier = MLPClassifier(hidden_layer_sizes=(m1, m2), activation='logistic', solver='adam',
                                   alpha=0.0001, batch_size='auto', learning_rate='constant',
                                   learning_rate_init=0.001, power_t=0.5, max_iter=200,
                                   shuffle=True, random_state=None, tol=0.0001, verbose=False,
                                   warm_start=False, momentum=0.9, nesterovs_momentum=True,
                                   early_stopping=True, validation_fraction=0.1, beta_1=0.9,
                                   beta_2=0.999, epsilon=1e-08)
        classifier.fit(ftr[:,1:-1],ftr[:,-1])
        ytr = classifier.predict(ftr[:, 1:-1])
        yval = classifier.predict(fval[:, 1:-1])
        ytest = classifier.predict(ftest[:, 1:])
        train_evalute = classification_report(ftr[:, -1], ytr)
        validation_evalute = classification_report(fval[:, -1], yval)
        #print train_evalute
        print validation_evalute
        #return classifier.predict_proba(ftest[:, 1:])
    #随机森林！
    def RF(self,n):
        ftr=self.traingSet
        fval=self.validationSet
        ftest=self.test
        fr=RandomForestClassifier(n_estimators=n,oob_score='True')
        #training!
        fr.fit(ftr[:,1:-1],ftr[:,-1])
        ytr = fr.predict(ftr[:, 1:-1])
        yval = fr.predict(fval[:, 1:-1])
        ytest = fr.predict(ftest[:, 1:])
        train_evalute = classification_report(ftr[:, -1], ytr)
        validation_evalute = classification_report(fval[:, -1], yval)
        #print train_evalute
        print validation_evalute
        #return classifier.predict_proba(ftest[:, 1:])
    # 随机森林输出！
    def RFOutPut(self, n):
        ftr=self.traingSet
        fval=self.validationSet
        ftest = self.test
        m,n=ftest.shape
        fr=RandomForestClassifier(n_estimators=n,oob_score='True')
        fr.fit(ftr[:, 1:-1], ftr[:, -1])
        ytr = fr.predict(ftr[:, 1:-1])
        yval=fr.predict(fval[:, 1:-1])
        ytest=fr.predict(ftest[:,1:])
        test_prob=fr.predict_proba(ftest[:,1:])
        test_output=np.zeros([m,2])
        test_output[:,0]=ftest[:,0]
        test_output[:,1]=test_prob[:,1]
        np.savetxt('result_201713.csv',test_output,delimiter=',')
        #return ftest

    # cross_validation:对这三个参数进行调优！
    #GBDT：调优！GridSearchCV:trainin不需要人为的划分了感觉还是不行！因为不知道他有没有考虑到正负样本的问题！
    def GBDT_CV(self,learningRate,nEstimators,maxDepth):
        ftr = self.traingSet
        m1,n1=ftr.shape
        fval = self.validationSet
        m2,n2=fval.shape
        gbdt=GradientBoostingClassifier(learning_rate=learningRate,n_estimators=nEstimators,max_depth=maxDepth)
        #training!
        gbdt.fit(ftr[:,1:-1],ftr[:,-1])
        ytr = gbdt.predict(ftr[:, 1:-1])
        ytr_prob=gbdt.predict_proba(ftr[:, 1:-1])
        yval = gbdt.predict(fval[:, 1:-1])
        yval_prob=gbdt.predict_proba(fval[:, 1:-1])
        #创建新的我们能够用于下一步KS测试的矩阵！
        #[user_id,lables,prob]
        #get train_ks
        train_ks=np.zeros([m1,3])
        train_ks[:,0]=ftr[:,0]#第一列是user_id
        train_ks[:,1]=ftr[:,-1]#第二列是lables
        #第三列是预测的概率！
        for i in range(m1):
            if(train_ks[i,1]==0):
                train_ks[i,2]=ytr_prob[i,0]
            else:
                train_ks[i,2]=ytr_prob[i,1]
        #get validation_ks
        validation_ks=np.zeros([m2,3])
        validation_ks[:,0]=fval[:,0]#第一列是user_id
        validation_ks[:,1]=fval[:,-1]#第二列是lables
        #第三列是预测的概率！
        for j in range(m2):
            if(validation_ks[j,1]==0):
                validation_ks[j, 2] = yval_prob[j, 0]
            else:
                validation_ks[j, 2] = yval_prob[j, 1]
        #print values
        ks_train=self.ks_score(train_ks)
        ks_val=self.ks_score(validation_ks)
        print(ks_train)
        print(ks_val)
        #return train_ks,validation_ks
    #cv完毕！找到三个参数的取值！输出！
    #(0.01,150,15)
    def GBDT_OutPut(self,learningRate,nEstimators,maxDepth):
        #ftr = self.traingSet_all
        ftr=self.traingSet
        ftest = self.test
        m, n = ftest.shape
        gbdt=GradientBoostingClassifier(learning_rate=learningRate,n_estimators=nEstimators,max_depth=maxDepth)
        #training!
        gbdt.fit(ftr[:,1:-1],ftr[:,-1])
        ytr = gbdt.predict(ftr[:, 1:-1])
        #ytest = gbdt.predict(ftest[:, 1:])
        test_prob=gbdt.predict_proba(ftest[:,1:])
        test_output=np.zeros([m,2])
        test_output[:,0]=ftest[:,0]
        test_output[:,1]=test_prob[:,1]
        np.savetxt('result_2017117_0.csv',test_output,delimiter=',')
        values = gbdt.feature_importances_
        #return ftest
        print values
    #想法测试！算了吧！
    def GBDT_CV_Test(self,learningRate,nEstimators,maxDepth):
        ftr = self.traingSet_all
        ftest = self.test
        gbdt = GradientBoostingClassifier(learning_rate=learningRate, n_estimators=nEstimators, max_depth=maxDepth)
        # training!
        gbdt.fit(ftr[:, 1:-1], ftr[:, -1])
        ytr = gbdt.predict(ftr[:, 1:-1])
        # ytest = gbdt.predict(ftest[:, 1:])
        train_evalute = classification_report(ftr[:, -1], ytr)
        values = gbdt.feature_importances_
        print values

    #Kolmogrov-Smirnov(KS) Score
    #先排好序噻！按最后一列从小到大整体排序！
    def sort_up(self,f):
        '''
        :param f:
        :return: l_in是set类型！list(l_in)变成list类型！
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
        #同时统计两类的个数！
        all_0=0
        all_1=0
        l_all=[]
        for i in f_new:
            l_all.append(i[-1])
            if(i[1]==0):
                all_0+=1
            else:
                all_1+=1
        #list 去重！返回set类型l_in
        #l_in=list(set(l_all)).sort()
        l_in=set(l_all)
        #print f_new
        #print list(l_in).sort()
        #print all_0
        #print all_1
        return f_new,l_in,all_0,all_1
    #计算KS评分！
    def ks_score(self,f):
        f_new,l_in,num_0,num_1=self.sort_up(f)
        #对l_in处理！
        l_new=np.sort(list(l_in))
        #print l_new
        #按照计算规则开始计算！
        #f:每一列的含义：[user_id,lable，prob]
        count_0=0
        count_1=0
        max_score=0
        for i in l_new:
            for j in f_new:
                if(j[2]==i):
                    if(j[1]==0):
                        count_0+=1
                    else:
                        count_1+=1
            #对每一个i都遍历一遍f_new
            #内循环遍历完事后！计算当前的差!
            temp=abs((count_0/(float(num_0)))-(count_1/(float(num_1))))
            if(temp>max_score):
                max_score=temp
        #外循环结束后返回score！
        return max_score
    #ks_score 测试：bingo!通过！
    def ks_score_test(self):
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
        Ks_score=self.ks_score(f_test)
        print Ks_score




