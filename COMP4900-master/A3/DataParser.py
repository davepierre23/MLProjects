from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords 
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk import SnowballStemmer
import string
import pandas as pd
import os 
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
class DataParser:
	def __init__(self):
		pass
		
	def import_reddit_dataset_train(self,path = '/hw2-reddit-classification/train.csv'):
		dataFrame=self.import_data_by_name(path)
		dataFrame = self.baseCleanData(dataFrame)
		return dataFrame
	def baseCleanData(self,data):
		dataFrame = data
		dataFrame["body"] = dataFrame["body"].apply(baseCleaner)
		return dataFrame

	
	def split_into_featuresMatrix_label(self,dataFrame):
		dataX = self.baseCleanData(dataFrame)
		dataX= dataX[dataX.columns[:-1]]
		dataX = dataX['body']
		dataX = dataX.to_numpy()
		y= dataFrame["subreddit"].to_numpy()
		return dataX , y
	
	def import_reddit_dataset_test(self,path = '/hw2-reddit-classification/test.csv'):
		return self.import_data_by_name(path)

	def import_data_by_name(self,dataPath):
		return pd.read_csv(os.getcwd()+dataPath)
	
	def featuresTfidNormalizedVector(self,trainData,testData):
		vectorizer = TfidfVectorizer(max_features=5000)
		vectors_train = vectorizer.fit_transform(trainData)
		X_train = vectors_train.toarray()
		vectorizer_test = vectorizer.transform(testData)
		X_test =vectorizer_test.toarray()  
	
		return X_train , X_test
		
	def featuresCountBinaryVector(self,trainData,testData):
		vectorizer = CountVectorizer(max_features=5000,binary=True)
		X_train = vectorizer.fit_transform(trainData).toarray()
		X_test = vectorizer.transform(testData).toarray()
		return X_train , X_test

#helper functions 
def stemmize(sentence):
  stemmer = SnowballStemmer("english")
  stemmize_tokens=[stemmer.stem(token) for token in word_tokenize(sentence)]
  return " ".join(stemmize_tokens)
  
def baseCleaner(sentence):
  func_cleaners=[removeStopwords,remove_punctuation,stemmize]
  sentence=sentence
  for clean_func in func_cleaners:
    sentence=clean_func(sentence)
  return sentence

def remove_punctuation(sentence):
  return sentence.translate(str.maketrans('', '', string.punctuation))

def removeStopwords(sentence):
  stop_words = set(stopwords.words('english')) 
  filtered_sentence = [w for w in word_tokenize(sentence)  if not w in stop_words] 
  return " ".join(filtered_sentence)