#this will provide stats for the two dataset

from  DataParser import DataParser as paser
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#this will provide stats for the two dataset
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np 


def get_histogram_Of_ClassLabels(train_pd,dataSetName):
    plt.figure(figsize=(10,5))
    sns.countplot(train_pd["subreddit"])
    #Added labels
    plt.title('Histogram of Class Labels for Reddict Text Classfication')
    plt.xlabel('Class Labels')
    plt.ylabel('Count')
    plt.savefig('Histogram_'+dataSetName+'.png',bbox_inches='tight')
    plt.show()

#used for Results
def graphAccuracy(group_names,group_data,dataSetName):
  
    #make graph for plot one 
    plt.figure(1)
    fig, ax = plt.subplots()
    plt.style.use('fivethirtyeight')
    ax.set(xlim=[0.5, 1], ylabel='Classfier Name', xlabel='Average Accuracy',title='Classifiers Accuracy on Reddit Data')
    graphResults(group_names,group_data,ax)
    plt.savefig('Accuracy_'+dataSetName+'.png',bbox_inches='tight')
    plt.show()
def graphResults(group_names,group_data,ax):

    barSize=2
    seperationSize=7
    print(group_names)
    #log Results
    logRegResults=group_data[2:4]
    logRegLabels = group_names[2:4]
    logRegXPos = np.arange(len(logRegResults)*barSize, step=barSize)

    print(logRegResults)
    
    #DEcision Tree results
    descionTreeResults=group_data[0:2]
    descionTreeLabels=group_names[0:2]
    descionTreeXpos = np.arange(len(descionTreeResults)*barSize,step=barSize) +    seperationSize
    
    #Bernouli results
    bnbResults = group_data[4:]
    bnbLabel = group_names[4:]
    bnbxPos = np.arange(len(bnbResults)) +    seperationSize*2


    for i in range(len(logRegXPos)):
      color_bar=assignFeatureColor(logRegLabels[i])
      bar = ax.barh(logRegXPos[i], logRegResults[i],barSize,  color=color_bar)
    for i in range(len(descionTreeLabels)):
      color_bar=assignFeatureColor(descionTreeLabels[i])
      bar = ax.barh(descionTreeXpos[i], descionTreeResults[i],barSize, color=color_bar)

    for i in range(len(bnbResults)):
      color_bar=assignFeatureColor(bnbLabel[i])
      bar = ax.barh(bnbxPos[i], bnbResults[i],barSize, color=color_bar)

    ax.set_yticks([logRegXPos[1],descionTreeXpos[1],bnbxPos[0]])
    ax.set_yticklabels(["Logistic Regression","Decison Tree","Bernoulli Naïve Bayes"])
    binary_label = mpatches.Patch(color='red', label='Bags of Words')
    tfIdf_label = mpatches.Patch(color='blue', label='TF-IDF')
    plt.legend(handles=[binary_label,tfIdf_label],title="Feature Vectorizer used",bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)



def graphRunTime(group_names,group_data,dataSetName):
    #make graph for plot 2
    plt.figure(1)
    fig, ax = plt.subplots()
    plt.style.use('fivethirtyeight')
    ax.set(xlim=[0,50], ylabel='Classfier Name-feature', xlabel='Average Runtime',title='Classifiers RunTime on Reddit Data')
    graphResults(group_names,group_data,ax)
    plt.savefig('RunTime_'+dataSetName+'.png',bbox_inches='tight')
    plt.show()


def assignClassiferLabel(classiferName):
  labels = ["Decision Tree", "Logistic Regression","Bernoulli Naïve Bayes"]
  for label in labels:
    if (classiferName.find(label)>0):
      return str(label)

def assignFeatureColor(classiferName):
  if (classiferName.find("BinaryCNT")>0):
    return "red"
  elif (classiferName.find("TFIDF")>0):
    return "blue"



#showing the top 10 words after removing the stop words and punctionations
def show_top_n_words_tf(trainData, n=10):
  #create a tfid vector with the data 
  vectorizer = TfidfVectorizer(max_features=5000)
  #create the words 
  words = vectorizer.fit_transform(trainData)
  #sum each colmns 
  sum_words = words.sum(axis=0) 
  #create a tupe of top words 
  term_freq = [(word, sum_words[0, index]) for word, index in  vectorizer.vocabulary_.items()]
  #sort the tope words
  term_freq =sorted(term_freq, key = lambda x: x[1], reverse=True)
  #get top n words 
  top_n_words=term_freq[:n]
  dict_n_words = createDictionaryOfTopWords(top_n_words)
  displayHistogramwords(dict_n_words,"Total TF-IDF value","Ten Highest TF-IDF Terms")
  

#showing the top 10 words after removing the stop words and punctionations
def show_top_n_words_cv(trainData, n=10):
  #count vectorizer
  vectorizer = CountVectorizer(max_features=5000,)
  #create the words 
  words = vectorizer.fit_transform(trainData)
  #sum each colmns 
  sum_words = words.sum(axis=0) 
  #create a tupe of top words 
  term_freq = [(word, sum_words[0, index]) for word, index in  vectorizer.vocabulary_.items()]
  #sort the tope words
  term_freq =sorted(term_freq, key = lambda x: x[1], reverse=True)
  #get top n words 
  top_n_words=term_freq[:n]
  dict_n_words = createDictionaryOfTopWords(top_n_words)
  displayHistogramwords(dict_n_words,"Term Frequency","Ten Highest Term Frequency")


def show_top_n_words_binary_cv(trainData, n=10):
  #count vectorizer
  vectorizer = CountVectorizer(max_features=5000,binary=True)
  #create the words 
  words = vectorizer.fit_transform(trainData)
  #sum each colmns 
  sum_words = words.sum(axis=0) 
  #create a tupe of top words 
  term_freq = [(word, sum_words[0, index]) for word, index in  vectorizer.vocabulary_.items()]
  #sort the tope words
  term_freq =sorted(term_freq, key = lambda x: x[1], reverse=True)
  #get top n words 
  top_n_words=term_freq[:n]
  dict_n_words = createDictionaryOfTopWords(top_n_words)
  displayHistogramwords(dict_n_words, "Number of documents containing term","Ten Most Common Terms")


def displayHistogramwords(common_words_dict,ylabel="values",title= "Top Ten Terms"):

  fig, ax = plt.subplots()
  plt.xlabel ('Terms')
  plt.ylabel (ylabel)
  plt.title(title)
 
  plt.bar(range(len(common_words_dict)), common_words_dict.values(),align='center')
  plt.xticks(range(len(common_words_dict)), common_words_dict.keys())
  plt.xticks(rotation=45)
  plt.savefig(title+'.png',bbox_inches='tight')
  plt.show()

def createDictionaryOfTopWords(wordlist):
  words_dict={}
  for word in wordlist:
    words_dict[word[0]]=word[1]
  return words_dict


if __name__ == "__main__":
    train_pd=parser.import_reddit_dataset_train()
    get_histogram_Of_ClassLabels()
    show_top_n_words_tf(train_pd['body'], 10)
    show_top_n_words_cv(train_pd['body'], 10)
    show_top_n_words_binary_cv(train_pd['body'], 10)


    
