
import matplotlib.pyplot as plt
from  CrossValidation import k_Fold_Validation
from DataParser import DataParser
from LogisiticRegression import LR
from Stats import *
import numpy as np
import time
def runTest(data_kFold_x, data_kFold_y):
    parser = DataParser()
    kFold= k_Fold_Validation()

    learningRates = np.arange(0.00, 0.12, 0.02)
    numIterations = 1000
    model_accuracies = []
    runTime =[]

    #create T models for cross Valiadation
    for rate in learningRates:
        model = LR(rate, numIterations)
        start = time.time()
        avgAccuracy = kFold.run_k_fold_cross_valid(data_kFold_x,data_kFold_y, model)
        #time it took to run cross validation
        end = time.time()
        runTime.append(end-start)
        model_accuracies.append(avgAccuracy)

    return learningRates ,model_accuracies, runTime

#running Hepatsis Test
#running Hepatsis Test
def runHeptatsis():
    parser = DataParser()
    dataSetName = "Hepatitis Dataset"

    plt.title(dataSetName)
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Accuracy ')

    #run test on original data
    data_kFold_x ,data_kFold_y, = parser.import_hepatitis()
    learningRates, modelResults, runTimes = runTest(data_kFold_x ,data_kFold_y)

    
    #run data on subselection of data
    data_kFold_x ,data_kFold_y = parser.import_hepatitis_selected_feat()
    learningRates, modelResults2,runTimes2  = runTest(data_kFold_x ,data_kFold_y)

    #graph results
    graphAccuracy(learningRates,modelResults,modelResults2,dataSetName)

    graphRunTime(learningRates,runTimes,runTimes2,dataSetName)



if __name__ == "__main__":
    runHeptatsis()


def runBankRuptcyTest():
    parser = DataParser()
    dataSetName = "Bankruptcy Dataset"
    
    #run test on original data
    data_kFold_x ,data_kFold_y, = parser.import_bankrupcy ()
    learningRates, modelResults, runTimes = runTest(data_kFold_x ,data_kFold_y)
    
    #run data on subselection of data
    data_kFold_x ,data_kFold_y = parser.import_bankrupcy_selected_feat()
    learningRates, modelResults2,runTimes2  = runTest(data_kFold_x ,data_kFold_y)

    #graph results
    graphAccuracy(learningRates,modelResults,modelResults2,dataSetName)
    
    graphRunTime(learningRates,runTimes,runTimes2,dataSetName)


    

if __name__ == "__main__":
    runBankRuptcyTest()



def main():
    runHeptatsis()
    runBankRuptcyTest()


if __name__ == "__main__":
    main()
