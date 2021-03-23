"""
A class that implement the k fold  cross validation algorithem
 
Input: 
      k= the number of folds to do 
"""
import numpy as np
import math
import copy

class k_Fold_Validation():
    def __init__(self, k=10):
        self.k = k
    def get_k_Partitions_And_Labels(self, X, Y, k):
        #from 1....k
        #add the ith row(example) to an list and step by k to get the next row
        #once it is done convert the list to an np array object
        #this solution will have each array have around the same size of examples instead of the last array having less then the other subset
        kPartitionFeatures = np.array([X[i::k] for i in range(k)])
        kPartitionLabels = np.array([Y[i::k] for i in range(k)])

        return kPartitionFeatures, kPartitionLabels

    def get_ith_traningSet_ValidSet(self, ithExperiment, kPartitionFeatures, kPartitionLabels, numRow, k):

        #the validation set is going to be on the number of experiment we have run
        validSet = copy.deepcopy(kPartitionFeatures[ithExperiment])
        validSetLabel = copy.deepcopy(kPartitionLabels[ithExperiment])

        #copy traning data
        numTrainingRow = numRow - validSet.shape[0]
        numDataColmns = validSet.shape[1]
        trainingSet = copy.deepcopy(kPartitionFeatures)
        trainingSetLabel = copy.deepcopy(kPartitionLabels)

        #remove the itth index subset because it was used for validation
        trainingSet = np.delete(kPartitionFeatures, ithExperiment, axis=0)
        trainingSetLabel = np.delete(kPartitionLabels, ithExperiment, axis=0)

        #concatenates all the elemenes into a single numpy arrray and then reshape the data to be the size required
        trainingSet = np.concatenate(trainingSet).ravel().reshape(
            numTrainingRow, numDataColmns)
        trainingSetLabel = np.concatenate(
            trainingSetLabel).ravel().reshape(numTrainingRow, 1)

        return trainingSet, trainingSetLabel, validSet, validSetLabel

    def getTestError(self, predictedLabels, trueLabels):

        #count the number of predicted labels the same as the true labels and divide it by the total numbers of labels
        errorRate =(predictedLabels != trueLabels).mean()
        print("the test error was " + str(errorRate))
        return errorRate


    def accul_eval(self, predictValues , trueLabel):
      return (predictValues == trueLabel ).mean()

    #used on one model 
    def run_k_fold_cross_valid(self,X, Y,model):
        k = self.k

        #splits the data in k parts
        kPartitionFeatures, kPartitionLabels = self.get_k_Partitions_And_Labels(
            X, Y, k)

        accuracies = np.zeros((k,))

        #run k experiments
        for ithExperiment in range(k):
            trainingSet, trainingSetLabel, validSet, validSetLabel = self.get_ith_traningSet_ValidSet(ithExperiment, kPartitionFeatures, kPartitionLabels, X.shape[0], k)

            #model trains on the traning data

            model.fit(trainingSet, trainingSetLabel)

            #model tests on validation set
            predictedLabels = model.predict(validSet)

            accuracies[ithExperiment] = self.accul_eval(predictedLabels, validSetLabel)

        #get the average error rate of this model
        avgAcc = accuracies.mean()
        print("The average accuracy", avgAcc)

        return avgAcc
        