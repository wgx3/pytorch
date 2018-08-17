#coding=utf-8
#chapter 2 code 2 
import Tumor_prediction_Data_processing as dataprocess
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import pandas as pd 
import numpy as np 
from sklearn.cross_validation import train_test_split
# 从sklearn.metrics里导入classification_report模块。
from sklearn.metrics import classification_report

# 创建特征列表。
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# 使用pandas.read_csv函数从互联网读取指定数据。
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names = column_names )

# 将?替换为标准缺失值表示。
data = data.replace(to_replace='?', value=np.nan)
# 丢弃带有缺失值的数据（只要有一个维度有缺失）。
data = data.dropna(how='any')

# 输出data的数据量和维度。
print(data.shape)


#split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size = 0.25,random_state = 33)

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#initialize the LogisticRegression and SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()

#use fit moudle in LogisticRegression
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)
sgdc.fit(X_train,y_train)
sgdc_y_predict = sgdc.predict(X_test)
# 调用LogisticRegression中的fit函数/模块用来训练模型参数。
lr.fit(X_train, y_train)
# 使用训练好的模型lr对X_test进行预测，结果储存在变量lr_y_predict中。
lr_y_predict = lr.predict(X_test)

# 调用SGDClassifier中的fit函数/模块用来训练模型参数s。
sgdc.fit(X_train, y_train)
# 使用训练好的模型sgdc对X_test进行预测，结果储存在变量sgdc_y_predict中。
sgdc_y_predict = sgdc.predict(X_test)

#Get the score of the testing  data by using logistic Regression model 
print('Accuracy of LR Classifier:', lr.score(X_test,y_test))
print classification_report(y_test,lr_y_predict,target_names=['Benign','Malignat'])