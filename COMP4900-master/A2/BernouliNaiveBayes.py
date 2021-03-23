import numpy as np

class BernoulliNaiveBayes():
    def __init__(self):
        self.isLaplace_smoothing = True

    
    def compute_class_prob(self,labels):
        #takes in the training set labels which is an numpy array

        #get the aavialable class labels options 
        self.class_names= np.unique(labels)

        #array to hold the class probabilties 

        class_probs ={}
        for class_k_name in self.class_names:
            #get all the rows that have the same value as class_k_name  and divide by the number of examples
            class_k_prob=(float(labels[labels==class_k_name].shape[0]))/(float(labels.shape[0]))

            #insert the probability of class k into the dictionary 
            class_probs[class_k_name]=np.log(class_k_prob)

        #save the list to the class
        self.class_prob = class_probs
   
    
    def compute_feature_prob(self,X_train,y_train):

        features_count=np.zeros((self.class_names.shape[0],X_train.shape[1]))
        features_prob=np.zeros((self.class_names.shape[0],X_train.shape[1]))
        for name in range(self.class_names.shape[0]):
            #create a vector of counts 
            examples_of_class_k = X_train[(self.class_names[name]==y_train)]
            num_class_k = examples_of_class_k.shape[0]
            feature_count_class_k = examples_of_class_k.sum(axis=0)
            features_count[name,:] = feature_count_class_k

            if self.isLaplace_smoothing:
                feature_count_class_k = feature_count_class_k+1
                num_class_k+=2
                features_prob[name,:]= feature_count_class_k / float(num_class_k)
            
               

        self.feature_count_=features_count
        self.features_prob= features_prob
        return features_count

        

    def fit(self,X,y):
        #X will be a sparse matrix of 5000 words where each row is an example anf colmns repreosnt a word of 5000 colmns 
        # each entry  is 1 or 0 representing if the word is present in the example 
        X = np.array(X)
        y= np.array(y)

        #estimate the class probabilties
        self.compute_class_prob(y)
        #estimate the feature  probabilties accoding to the class names
        self.compute_feature_prob(X,y)
    

    def predict(self,X):
      # return predictions corresponding to the data X
      predicts=[self.predict_sample(i) for i in X]
      return np.array(predicts)

 
    def predict_sample(self,testPoint):
        #uses the log likelihood to predict samples of data

        #contins the lists of all the probabilties that are likely 
        class_probs=[]

        #do for all classes available
        for class_k_name in range(len(self.class_names)):
          feature_likehood=0
          #compute the likelihood of the features
          feat_prob=np.log(self.features_prob[class_k_name])
          x_i_j =testPoint*feat_prob

          not_feat_prob=np.log(1-self.features_prob[class_k_name]) 
          not_x_i_j= (1-testPoint) * not_feat_prob
          
          feature_likehood+=x_i_j.sum()+not_x_i_j.sum()
        
          #add the class proabbailties to the total 
          class_prob=feature_likehood + self.class_prob[self.class_names[class_k_name]]
          class_probs.append(class_prob)
        
        #returns the the class label with the highest probabilties 
        return self.class_names[np.argmax(class_probs)]