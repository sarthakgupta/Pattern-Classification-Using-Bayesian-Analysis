# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:01:55 2018

@author: Sarthak
"""

#from PIL import Image

#im = Image.open("C:\\sarthak\\CEDT IISC\\Semester 2\\Basics of Signal Processing\\Images\\Digit4_1.png")

import scipy.io
import numpy as np

mat=scipy.io.loadmat("Images\\matlab.mat")
mat_1=np.matrix(mat['I2'])

posterior=[]
x_test=np.matmul(mat_1,V1[:f_sel,:].T).T
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

print("Predicted Digit::",np.argmax(posterior))