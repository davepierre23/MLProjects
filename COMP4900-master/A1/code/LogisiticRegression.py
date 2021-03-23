"""
A class that represents a Logistic Regression Model
using cross entropy function 
Input: 
    learning_rate: The amount that the weights are updated during training
    training_iterations: number of times we should train our model 

"""

import pandas as pd 
import numpy as np 
class LR():
  def __init__(self,learning_rate,training_iteration,epilson=0.02):
    #each model will be defined by the learning rate and number of training_iteration
    self.k = training_iteration
    self.learnin_rate = learning_rate
    self.weights =[] #intiaize weights to be empty
    self.epilson= epilson

  def sigmoid(self,x):
    # 1/ 1 + e ^-x     
    # where x =wTx 
    return 1 / (1 + np.exp(-x,dtype=np.float128))

  def computeSigmoidInput(self,xi):
    # wTx 
    return np.dot(self.weights.T,xi)

    
  def fit(self,X,Y):
    #This model will learned based on the cross entropy function 
    intercept = np.ones((X.shape[0],1))
    X = np.concatenate((intercept,X),axis=1)
    
    self.weights= np.zeros(X.shape[1],dtype=np.float128) 

    for iter in range(self.k):
      gradient=0
      for i in range(X.shape[0]):

        xi=np.array(X[i],dtype=np.float128 )# xi represents an the ith exaample in our traning example 
        yi=np.array(Y[i],dtype=np.float128) # yi represents an the ith exaample in our label example 

        sigmoidInput= self.computeSigmoidInput(xi)# z= (wT * xi)
        predicted_output= self.sigmoid(sigmoidInput) # sigmoid(z)
        
        error = yi-predicted_output #computing the error error=y- sigmoid(z)
        
        gradient = gradient + (xi*error) # addin
        

      oldWeight = self.weights
      # updating the weights for the model
      self.weights = self.weights + (self.learnin_rate * (gradient))
      norms = np.linalg.norm(self.weights - oldWeight)**2
      # end when there almost no difference in the new weights and old one
      if(norms<self.epilson):
          return

   
  def predict(self,data):
    #model is making a new prediction
    #calculating the a probabilties vector for a given new feature vector data
    intercept = np.ones((data.shape[0],1))
    data = np.concatenate((intercept,data),axis=1)
    prVector=[self.sigmoid(self.computeSigmoidInput(row)) for row in data] #sigmoid(wT * x_new )  

    # functions to determine based off the probabilty what class does the feature fall in
    classify= lambda x: 1 if x >= 0.5 else 0 #our decison boundary will be places at 0.5
    predictions=[classify(xi) for xi in prVector]
    return predictions