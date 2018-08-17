#chapter 2 
#data process
import pandas as pd 
import numpy as np 
from sklearn.cross_validation import train_test_split

def dataprocessing():

	column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
	#load the data from internet by pandas.read_csv
	data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
	#print(data)
	#replace the missing data by ?
	data = data.replace(to_replace='? ',value = np.nan)
	data = data.dropna(how = 'any')
	#data = data.dropna()
	print(data.shape)

	#split the data for training and testing
	X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size = 0.25,random_state = 33)
	print(y_train.value_counts())
	print(y_test.value_counts())
	print("_______________________________________________")

#def main():
	#dataprocessing()

#if __name__ == "__main__":
	#main()