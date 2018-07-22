"""
Created on Sat Feb 17 11:50:50 2018

@author: Sarthak
"""

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from time import time

def normal_func(mean,inv_var,slog_det_var,x):
    exponential=(np.dot(np.dot((x-mean).T,inv_var),(x-mean)))/(-2)
    if(exponential<50):
    #y=math.exp((np.dot(np.dot((x-mean).T,np.linalg.inv(var)),(x-mean)))/(-2))/math.sqrt(logdet)
        y=math.exp(exponential)/math.sqrt(slog_det_var)
    else:
        y=2.8677702753877523e+21
        
    return y

#Reading Test Data
data_set_test=pd.read_csv("C:\\sarthak\\CEDT IISC\\Data Sets\\mnist_test_10000.csv")
test_width=data_set_test.shape[1]
test_header=[]
test_header.append('Label')
for i in range(0, test_width-1):
    test_header.append(str("Val")+str(i))
data_set_test=pd.read_csv("C:\\sarthak\\CEDT IISC\\Data Sets\\mnist_test_10000.csv",names=test_header)    
       
#Reading Train Data(Using the same header as test data)
data_set_train=pd.read_csv("C:\\sarthak\\CEDT IISC\\Data Sets\\mnist_train_60000.csv",names=test_header)
#############################################################################
#Comment the following line if do not want to add test data to training data#
#############################################################################
#data_set_train=pd.concat([data_set_train,data_set_test])
#############################################################################
data_set_train_1=data_set_train

print("Feature Engg starts::")
start=time()

data_set_train_features=np.matrix(data_set_train.iloc[:,1:])
data_set_test_features=np.matrix(data_set_test.iloc[:,1:])

#f_sel=10
acc_list=[]
f_sel_list=[]
f_sel_max=201
f_sel_opt=35

U1, S1, V1= np.linalg.svd(data_set_train_features, full_matrices=False)
S1=S1.reshape(-1,1)

#UnComment the following to perform Cross Validation
#for f_sel in range(5,f_sel_max,5):
for f_sel in range(f_sel_opt,f_sel_opt+1,5):    
    f_sel_list.append(f_sel)
    print("f_sel::",f_sel)
    ############################################################################
    ###################################Using Normal SVD#########################
    ############################################################################
#    print("Normal SVD Starts::")
#    U1, S1, V1= np.linalg.svd(data_set_train_features, full_matrices=False)
#    S1=S1.reshape(-1,1)
    
    data_set_train_features_reduced=np.matmul(U1[:,:f_sel],np.diag(S1[:f_sel,0]).T)
    data_set_test_features_reduced=np.matmul(data_set_test_features,V1[:f_sel,:].T)
#    print("Normal SVD Ends::")
    ############################################################################
    
    ############################################################################
    ###################################Using Truncate SVD#######################
    ############################################################################
#    print("Truncate SVD Starts::")
#    svd = TruncatedSVD(n_components=f_sel, n_iter=7, random_state=42)
#    svd.fit(data_set_train_features)
#    
#    data_set_train_features_reduced=svd.transform(data_set_train_features)
#    data_set_test_features_reduced=svd.transform(data_set_test_features)
#    print("Truncate SVD Ends::")
    ############################################################################
    
    data_set_train_reduced=np.concatenate((np.matrix(data_set_train.iloc[:,0]).T,data_set_train_features_reduced),axis=1)
    data_set_test_reduced=np.concatenate((np.matrix(data_set_test.iloc[:,0]).T,data_set_test_features_reduced),axis=1)
    
    data_set_train_1 = pd.DataFrame(data_set_train_reduced)
    data_set_test_1  = pd.DataFrame(data_set_test_reduced)
    
    data_set_class0=data_set_train_1[data_set_train_1[0]==0]
    data_set_class1=data_set_train_1[data_set_train_1[0]==1]
    data_set_class2=data_set_train_1[data_set_train_1[0]==2]
    data_set_class3=data_set_train_1[data_set_train_1[0]==3]
    data_set_class4=data_set_train_1[data_set_train_1[0]==4]
    data_set_class5=data_set_train_1[data_set_train_1[0]==5]
    data_set_class6=data_set_train_1[data_set_train_1[0]==6]
    data_set_class7=data_set_train_1[data_set_train_1[0]==7]
    data_set_class8=data_set_train_1[data_set_train_1[0]==8]
    data_set_class9=data_set_train_1[data_set_train_1[0]==9]
    
    max_val=1
    data_set_test_mat = np.matrix(data_set_test_1.iloc[:,:])
    print("Time taken by Feature engg::", time()-start)
    print("Feature Engg ends::")
    
    max_examples=np.max((data_set_class0.shape[0],data_set_class1.shape[0],data_set_class2.shape[0],
                     data_set_class3.shape[0],data_set_class4.shape[0],data_set_class5.shape[0],
                     data_set_class6.shape[0],data_set_class7.shape[0],data_set_class8.shape[0],
                     data_set_class9.shape[0]))
    
    prior_class0=len(data_set_class0)/len(data_set_train_1)
    prior_class1=len(data_set_class1)/len(data_set_train_1)
    prior_class2=len(data_set_class2)/len(data_set_train_1)
    prior_class3=len(data_set_class3)/len(data_set_train_1)
    prior_class4=len(data_set_class4)/len(data_set_train_1)
    prior_class5=len(data_set_class5)/len(data_set_train_1)
    prior_class6=len(data_set_class6)/len(data_set_train_1)
    prior_class7=len(data_set_class7)/len(data_set_train_1)
    prior_class8=len(data_set_class8)/len(data_set_train_1)
    prior_class9=len(data_set_class9)/len(data_set_train_1)
    
    #################################################
    #Using ML Estimate to determine mean and variance
    #################################################    
    print("ML Estimation Starts::")
    start=time()
    data_set_class0_temp= np.matrix(data_set_class0.iloc[:,1:])/max_val
    data_set_class1_temp= np.matrix(data_set_class1.iloc[:,1:])/max_val
    data_set_class2_temp= np.matrix(data_set_class2.iloc[:,1:])/max_val
    data_set_class3_temp= np.matrix(data_set_class3.iloc[:,1:])/max_val
    data_set_class4_temp= np.matrix(data_set_class4.iloc[:,1:])/max_val
    data_set_class5_temp= np.matrix(data_set_class5.iloc[:,1:])/max_val
    data_set_class6_temp= np.matrix(data_set_class6.iloc[:,1:])/max_val
    data_set_class7_temp= np.matrix(data_set_class7.iloc[:,1:])/max_val
    data_set_class8_temp= np.matrix(data_set_class8.iloc[:,1:])/max_val
    data_set_class9_temp= np.matrix(data_set_class9.iloc[:,1:])/max_val
    
    expected_mean_class0= data_set_class0_temp.mean(0)
    expected_mean_class1= data_set_class1_temp.mean(0)
    expected_mean_class2= data_set_class2_temp.mean(0)
    expected_mean_class3= data_set_class3_temp.mean(0)
    expected_mean_class4= data_set_class4_temp.mean(0)
    expected_mean_class5= data_set_class5_temp.mean(0)
    expected_mean_class6= data_set_class6_temp.mean(0)
    expected_mean_class7= data_set_class7_temp.mean(0)
    expected_mean_class8= data_set_class8_temp.mean(0)
    expected_mean_class9= data_set_class9_temp.mean(0)
    
    feature_len=data_set_class0_temp.shape[1]
    var_class0=np.cov(data_set_class0_temp.T)
    var_class1=np.cov(data_set_class1_temp.T)
    var_class2=np.cov(data_set_class2_temp.T)
    var_class3=np.cov(data_set_class3_temp.T)
    var_class4=np.cov(data_set_class4_temp.T)
    var_class5=np.cov(data_set_class5_temp.T)
    var_class6=np.cov(data_set_class6_temp.T)
    var_class7=np.cov(data_set_class7_temp.T)
    var_class8=np.cov(data_set_class8_temp.T)
    var_class9=np.cov(data_set_class9_temp.T)
    
    inv_var_class0=np.linalg.inv(var_class0)
    inv_var_class1=np.linalg.inv(var_class1)
    inv_var_class2=np.linalg.inv(var_class2)
    inv_var_class3=np.linalg.inv(var_class3)
    inv_var_class4=np.linalg.inv(var_class4)
    inv_var_class5=np.linalg.inv(var_class5)
    inv_var_class6=np.linalg.inv(var_class6)
    inv_var_class7=np.linalg.inv(var_class7)
    inv_var_class8=np.linalg.inv(var_class8)
    inv_var_class9=np.linalg.inv(var_class9)
    
    det_var_class0=np.linalg.slogdet(var_class0)[1]
    det_var_class1=np.linalg.slogdet(var_class1)[1]
    det_var_class2=np.linalg.slogdet(var_class2)[1]
    det_var_class3=np.linalg.slogdet(var_class3)[1]
    det_var_class4=np.linalg.slogdet(var_class4)[1]
    det_var_class5=np.linalg.slogdet(var_class5)[1]
    det_var_class6=np.linalg.slogdet(var_class6)[1]
    det_var_class7=np.linalg.slogdet(var_class7)[1]
    det_var_class8=np.linalg.slogdet(var_class8)[1]
    det_var_class9=np.linalg.slogdet(var_class9)[1]
    print("Time taken by ML Estimation::", time()-start)
    print("ML Estimation Ends::")
    
    #################################################
    #Implementing Bayes classifier
    #0-1 Loss Function
    #################################################
    print("Implementing Bayesian Model over Test Data")
    err=0
    start=time()
    for i3 in range (0, len(data_set_test_mat)):
        posterior=[]
        x_test=(data_set_test_mat[i3,1:].T)/max_val
        posterior.append(normal_func(expected_mean_class0.T,inv_var_class0,det_var_class0,x_test)*prior_class0)
        posterior.append(normal_func(expected_mean_class1.T,inv_var_class1,det_var_class1,x_test)*prior_class1)
        posterior.append(normal_func(expected_mean_class2.T,inv_var_class2,det_var_class2,x_test)*prior_class2)
        posterior.append(normal_func(expected_mean_class3.T,inv_var_class3,det_var_class3,x_test)*prior_class3)
        posterior.append(normal_func(expected_mean_class4.T,inv_var_class4,det_var_class4,x_test)*prior_class4)   
        posterior.append(normal_func(expected_mean_class5.T,inv_var_class5,det_var_class5,x_test)*prior_class5)
        posterior.append(normal_func(expected_mean_class6.T,inv_var_class6,det_var_class6,x_test)*prior_class6)
        posterior.append(normal_func(expected_mean_class7.T,inv_var_class7,det_var_class7,x_test)*prior_class7)
        posterior.append(normal_func(expected_mean_class8.T,inv_var_class8,det_var_class8,x_test)*prior_class8)
        posterior.append(normal_func(expected_mean_class9.T,inv_var_class9,det_var_class9,x_test)*prior_class9)    
    
        if(np.argmax(posterior)!=data_set_test_mat[i3,0]):
                err=err+1
        #print("itr::",i3,"Class Determined::", np.argmax(posterior), "Actual Class::",data_set_test_mat[i3,0], "Error::",err)            
    acc = 1-err/len(data_set_test)
    acc_list.append(acc)
    print("time::",time()-start,"f_sel::",f_sel,"Accuracy::",acc)        
        
plt.figure(1)
plt.plot(f_sel_list,acc_list,'b',label='Validation Accuracy (MNIST)')
plt.legend(loc='upper left',fontsize=15)
plt.ylabel('Accuracy',fontsize=15)
plt.xlabel('Dimensionality',fontsize=15)
plt.ylim((0.6,1))
plt.show()

      
        
        
        
        