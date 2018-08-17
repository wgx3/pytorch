# coding=utf-8
import pandas as pd
#调用pandas工具包的read_csv函数，传入训练文件地址参数，获得返回数据并存至变量df_train
df_train = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-train.csv')

#调用pandas工具包的read_csv函数，传入测试文件地址参数，获得返回数据并存至变量df_test
df_test = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-test.csv')

#选取'Clump Thickness 与 Cell Size 作为特征，构建测试集中的政府分类样本'
df_test_negative = df_test.loc[df_test['Type'] == 0] [['Clump Thickness','Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1] [['Clump Thickness','Cell Size']]

#import the pyplot in matplotlib tools named plt
import matplotlib.pyplot as plt
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')

#the discription of xlabel and ylabel
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

#import numpy tools named as np
import numpy as np
intercept = np.random.random([1]) 
coef = np.random.random([2])
lx = np.arange(0,12)
ly = (-intercept-lx * coef[0])/coef[1]
plt.plot(lx,ly,c = 'yellow')

plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')

#the discription of xlabel and ylabel
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

#import sklearn
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10],df_train['Type'][:10])
print('Testing accuracy(10 training samples:',lr.score(df_test[['Clump Thickness', 'Cell Size'],df_test['Type']))
intercept = lr.intercept_
coef = lr.coef_[0,:]
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c = 'blue')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')

#the discription of xlabel and ylabel
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()