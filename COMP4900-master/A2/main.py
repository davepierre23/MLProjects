
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from DataParser import DataParser
from BernouliNaiveBayes import *
from Stats import *
from sklearn.model_selection import KFold
import numpy as np
import time


def runKFoldFeatures(X,y,model,featureExtracter):

    model_accuracies = []
    runTime =[]

    # Create instance of KFold and give it number of split, and to shuffle before splitting
    kf = KFold(n_splits = 10, random_state=1, shuffle=True)

    # for each fold
    for train_index, test_index in kf.split(X):
        X_train=X[train_index]
        X_test=X[test_index]
        X_train, X_test= featureExtracter(X_train,X_test)
    
        # start the timer 
        start = time.time()

        # fitting the model
        model.fit(X_train, y[train_index])

        # getting predictions from the model
        predicted_set = model.predict(X_test)

        # stop the timer
        end = time.time()

        # get the accuracy of the model in this fold
        accuracy = accuracy_score(y[test_index], predicted_set)

        # save the accuracy and runtime
        model_accuracies.append(accuracy)
        runTime.append(end - start)
        

    avg_acc=np.array(model_accuracies).mean()
    avg_runTime = np.array(runTime).mean()
    print("Avg Accuracy  "+str(avg_acc))
    print("Avg Runtime "+ str(avg_runTime))
    return avg_acc, avg_runTime
    



def runTest():
    parser = DataParser()
    train_pd=parser.import_reddit_dataset_train()
    trainData, labels = parser.split_into_featuresMatrix_label(train_pd)
    dataSetName = "Reddit Text Dataset"

    print("Starting Reddit Test")

    #getting data from local variables 
    X_data = trainData
    y= labels

    runTimeData = {}
    accuracyData= {}



    names = ["Decision Tree-TFIDF","Decision Tree-BinaryCNT"]
    accuracyData[names[0]], runTimeData[names[0]] = runKFoldFeatures(X_data,y,DecisionTreeClassifier(),parser.featuresTfidNormalizedVector)
    accuracyData[names[1]], runTimeData[names[1]]=  runKFoldFeatures(X_data,y,DecisionTreeClassifier(),parser.featuresCountBinaryVector)
    names.append("Logistic Regression-TFIDF")
    accuracyData[names[2]], runTimeData[names[2]] = runKFoldFeatures(X_data,y,LogisticRegression(max_iter=1000, C=1),parser.featuresTfidNormalizedVector)
    names.append("Logistic Regression-BinaryCNT")
    accuracyData[names[3]], runTimeData[names[3]]= runKFoldFeatures(X_data,y,LogisticRegression(max_iter=1000, C=1),parser.featuresCountBinaryVector)
    names.append("Bernoulli Naïve Bayes-BinaryCNT")
    accuracyData[names[4]], runTimeData[names[4]]= runKFoldFeatures(X_data,y,BernoulliNaiveBayes(),parser.featuresCountBinaryVector)
    accuracy_data = list(accuracyData.values())
    runTime_data = list(runTimeData.values())
    group_names = list(accuracyData.keys()) 

    return group_names ,accuracy_data, runTime_data , dataSetName
if __name__ == "__main__":
  group_names, accuracy_data, runTime_data , dataSetName = runTest()
  graphAccuracy(group_names,accuracy_data,dataSetName)
  graphRunTime(group_names,runTime_data,dataSetName)
  print("Runtime of test:")
  for i in range(len(group_names)):
    print(group_names[i]+" : " +str(runTime_data[i]))
  print()
  print("Accuracy of test:") 
  for i in range(len(group_names)):
    print(group_names[i]+" : " +str(accuracy_data[i]))