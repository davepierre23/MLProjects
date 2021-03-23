#this will provide stats for the two dataset

import DataParser as parser
import matplotlib.pyplot as plt
import seaborn as sns


def get_histogram_Of_ClassLabels(data,name):
    dataframe= parser.get_data_frame_bankrupcy()
    print(dataframe)

    print(dataframe['ClassLabel'])


    #plt.hist(dataframe['ClassLabel'], color='blue', edgecolor='black',bins =2)
    sns.distplot(dataframe['ClassLabel'],hist=True,kde=False,bins=2,color='blue',hist_kws={'edgecolor':'black'})

    #Added labels
    plt.title('Historgram of Class Labels for Bankrupcy')
    plt.xlabel('Class Label')
    plt.ylabel('Count')



    plt.show()

#used for Results
def graphAccuracy(learningRates,modelResults,modelResults2,dataSetName):

    #make graph for plot one 
    plt.figure(1)
    plt.title("Accuracy on "+dataSetName)
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Accuracy')
    plt.plot(learningRates, modelResults, label ="Original Features",color='blue',marker='o')
    plt.plot(learningRates, modelResults2,label="Selected Features", color='red',linestyle='--',marker='o')
    plt.legend()
    plt.savefig('Accuracy_'+dataSetName+'.png',bbox_inches='tight')
    plt.show()


def graphRunTime(learningRates,runTimes,runTimes2,dataSetName):
    #make graph for plot 2
    plt.figure(2)
    plt.title("Runtime on "+dataSetName)
    plt.xlabel('Learning Rate')
    plt.ylabel('Time of Accuracy')
    plt.plot(learningRates, runTimes, label ="Original Features",color='blue',marker='o')
    plt.plot(learningRates, runTimes2,label="Selected Features", color='red',linestyle='--',marker='o')
    plt.legend()
    plt.savefig('RunTime_'+dataSetName+'.png',bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    get_histogram_Of_ClassLabels()


    
