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
    def __init__(self,ratio):
        self.Dp=Q1.MyDataProcess()
        self.traingSet,self.validationSet=self.Dp.posAndNegtive1(ratio)
        self.test=self.Dp.testingSet
        self.traingSet_all=self.Dp.trainAllUsing(ratio)
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
        fval = self.validationSet
        ftest = self.test
        gbdt=GradientBoostingClassifier(learning_rate=learningRate,n_estimators=nEstimators,max_depth=maxDepth)
        #training!
        gbdt.fit(ftr[:,1:-1],ftr[:,-1])
        ytr = gbdt.predict(ftr[:, 1:-1])
        yval = gbdt.predict(fval[:, 1:-1])
        #ytest = gbdt.predict(ftest[:, 1:])
        train_evalute = classification_report(ftr[:, -1], ytr)
        validation_evalute = classification_report(fval[:, -1], yval)
        values=gbdt.feature_importances_
        print validation_evalute
        print values
    #cv完毕！找到三个参数的取值！输出！
    #(0.01,150,15)
    def GBDT_OutPut(self,learningRate,nEstimators,maxDepth):
        ftr = self.traingSet_all
        ftest = self.test
        m, n = ftest.shape
        gbdt=GradientBoostingClassifier(learning_rate=learningRate,n_estimators=nEstimators,max_depth=maxDepth)
        #training!
        gbdt.fit(ftr[:,1:-1],ftr[:,-1])
        ytr = gbdt.predict(ftr[:, 1:-1])
        ytest = gbdt.predict(ftest[:, 1:])
        test_prob=gbdt.predict_proba(ftest[:,1:])
        test_output=np.zeros([m,2])
        test_output[:,0]=ftest[:,0]
        test_output[:,1]=test_prob[:,1]
        np.savetxt('result_201716_1.csv',test_output,delimiter=',')
        #return ftest
    #想法测试！算了吧！
    def GBDT_CV_Test(self,learningRate,nEstimators,maxDepth):
        ftr = self.traingSet
        fval = self.validationSet
        m,n=fval.shape
        gbdt=GradientBoostingClassifier(learning_rate=learningRate,n_estimators=nEstimators,max_depth=maxDepth)
        #training!
        gbdt.fit(ftr[:,1:-1],ftr[:,-1])
        ytr = gbdt.predict(ftr[:, 1:-1])
        yval = gbdt.predict(fval[:, 1:-1])
        #ytest = gbdt.predict(ftest[:, 1:])
        train_evalute = classification_report(ftr[:, -1], ytr)
        validation_evalute = classification_report(fval[:, -1], yval)
        print validation_evalute
        yval_prob = gbdt.predict_proba(fval[:, 1:-1])
        #new features!
        fval_new=np.zeros([m,n+1])
        fval_new[:,-1]=fval[:,-1]
        fval_new[:,-2]=yval_prob
        fval_new[:,:-2]=fval[:,:-1]
        #reTraining!
        gbdt = GradientBoostingClassifier(learning_rate=learningRate, n_estimators=nEstimators, max_depth=maxDepth)
        gbdt.fit(ftr[:, 1:-1], ftr[:, -1])

