#coding=utf-8
#chapter 2 
#code 5
#This moudle is one of the Ensemble moudles named Random Forest Classfier
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#print(titanic)
X=titanic[['pclass','age','sex']]
y=titanic[['survived']]
X['age'].fillna(X['age'].mean(),inplace=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)
vec=DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.transform(X_test.to_dict(orient='record'))
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_pred=dtc.predict(X_test)
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred=rfc.predict(X_test)
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred=gbc.predict(X_test)
print('The accuracy of decision tree is ',dtc.score(X_test,y_test))
print('The accuracy of random forest tree is ',rfc.score(X_test,y_test))
print('The accuracy of GBDT is ',gbc.score(X_test,y_test))
