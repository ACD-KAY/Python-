# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 09:41:46 2018

@author: asus
"""

#导入iris数据加载器
from sklearn.datasets import load_iris
#使用加载器读取数据并存入变量iris
iris=load_iris()

#查验数据规模
#print(iris.data.shape)

#查看数据说明
#print(iris.DESCR)

#对数据进行随机分割
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)

#使用K近邻分类器对数据进行类别预测
#导入标准化模块
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


#对训练数据的测试的特征数据进行标准化
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

#使用K近邻分类器对测试数据进行类别预测，预测结果保存在y_predict中
knc=KNeighborsClassifier()
knc.fit(X_train,y_train)
y_predict=knc.predict(X_test)

#对K近邻分类器数据的预测性能进行评估
print('The accuracy of K-Nearest Neighbor Classifier is',knc.score(X_test,y_test))
#详情分析
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=iris.target_names))