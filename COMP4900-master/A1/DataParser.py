"""
A class that will get the data for both datasets
"""
import pandas as pd
import os
from math import floor
import numpy as np
import sklearn.linear_model
from sklearn import preprocessing

#values for Lasso regularization
ALPHA=0.001
class DataParser:
	def __init__(self):
		pass
		
	def import_bankrupcy(self,path = '/Data/bankrupcy.csv'):
		return self.import_data_by_name(path)


	def import_hepatitis(self,path = '/Data/hepatitis.csv'):
		return self.import_data_by_name(path)

	def import_bankrupcy_selected_feat(self,path = '/Data/bankrupcy.csv'):

		return self.import_selected_features(path)

	def import_hepatitis_selected_feat(self,path = '/Data/hepatitis.csv'):
		return self.import_selected_features(path)

	def import_data_by_name(self,dataPath):
		dataFrame = pd.read_csv(os.getcwd()+dataPath)
		dataFrame = dataFrame.sample(frac=1) # randomize dataset

		#Data setup feature1 + feature 2+......feature m + Label
		#keep all colmns except for the last one 
		featuresMatrix = dataFrame[dataFrame.columns[:-1]].to_numpy()

		#retrieve only the last column and turn it into a nunmpy object
		labels= dataFrame['ClassLabel'].to_numpy()
		featuresMatrix = self.normalization(featuresMatrix)
		return featuresMatrix , labels


	#peform normalization technique to out feature dataset
	def normalization(self,featuresMatrix):
		scaler = preprocessing.StandardScaler()
		#substract the mean of each colmn and divide by the standard deviation 
		scaled_values = scaler.fit_transform(featuresMatrix) 
		return scaled_values

	"""
	function that return selected features based on :
	L1 (Lasso) Regression: Choose features not weighted 0
	"""
	def import_selected_features(self,datapath):
		dataFrame = pd.read_csv(os.getcwd()+datapath)
		dataFrame = dataFrame.sample(frac=1) # randomize dataset
		dataFrameX = dataFrame.drop('ClassLabel', 1)
		
		dataFrameY = dataFrame['ClassLabel']
		dataColsNames = dataFrame.columns.to_list()
		linearModelOfData = sklearn.linear_model.Lasso(alpha=ALPHA)
		linearModelOfData.fit(dataFrameX, dataFrameY)
		#Print all items features not weighted 0

		#print("Selected Features: \n")
		nameOfSelectedFeatures = [] #List for the selected features
		i = 0
		while i < len(linearModelOfData.coef_):
			if(linearModelOfData.coef_[i] != 0):
				#print(str(dataColsNames[i]) + ": " + str(linearModelOfData.coef_[i]))
				nameOfSelectedFeatures.append(dataColsNames[i])#added colmns names where the feature is weighted not zerp
			i = i+1
		nameOfSelectedFeatures.append('ClassLabel')# add the Class label

		#selected features where they are 
		dfSelectFeatures= dataFrame[nameOfSelectedFeatures]

		#Data setup feature1 + feature 2+......feature m + Label
		#keep all colmns except for the last one 
		featuresMatrix = dfSelectFeatures[dfSelectFeatures.columns[:-1]].to_numpy()

		#retrieve only the last column and turn it into a nunmpy object
		labels= dataFrame['ClassLabel'].to_numpy()
		
		featuresMatrix = self.normalization(featuresMatrix)

		return featuresMatrix , labels
