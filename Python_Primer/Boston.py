# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 22:37:38 2018

@author: asus
"""

from sklearn.datasets import load_boston
boston=load_boston()
#print (boston.DESCR)
from sklearn.cross_validation import train_test_split
import numpy as np
X=boston.data
y=boston.target

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)


#分析回归目标值的差异
print ('The max target value is',np.max(boston.target))
print ('The min target value is',np.min(boston.target))
print ('The average target value is',np.mean(boston.target))

#训练与测试数据标准化模块
from sklearn.preprocessing  import StandardScaler

ss_X=StandardScaler()
ss_Y=StandardScaler()

#分别对训练和测试数据的特征以及目标值进行标准化处理
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_Y.fit_transform(y_train.reshape(-1,1))
y_test=ss_Y.transform(y_test.reshape(-1,1))


#使用线性回归模型LinearRegression和SGDRegressor分别对美国波士顿地区房价进行预测
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
#使用训练数据进行参数估计
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)


from sklearn.linear_model import SGDRegressor
sgdc=SGDRegressor()
sgdc.fit(X_train,y_train)
sgdc_y_predict=sgdc.predict(X_test)

print('The value of default measurement of LinearRegressin is',lr.score(X_test,y_test))

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('The value of R-squared of LinearRegression is',r2_score(y_test,lr_y_predict))
#inverse_transform标准化还原
print('The mean squared error of LinearRegression is',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(lr_y_predict)))

print('The mean absoluate error of LinearRegression is',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(lr_y_predict)))


#使用SGDRegression模型自带的评估模块，并输出评估结果
print ('The value of default measurement of SGDRegressior is',sgdc.score(X_test,y_test))

#使用r2_score
print ('The value of R-squared of SGDRegressior is',r2_score(y_test,sgdc_y_predict))

#使用mean_squared_error
print ('The mean squared error of SGDRegressor is ',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(sgdc_y_predict)))

#使用mean_absolute_error
print ('The mean absolute error SGDRegressor is',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(sgdc_y_predict)))


