import os
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict


def gather_data():
	test_dir='C:\\Users\\pl\\Downloads\\20news-bydate\\20news-bydate-test'
	train_dir='C:\\Users\\pl\\Downloads\\20news-bydate\\20news-bydate-train'
	list_newsgroups=[newsgroup for newsgroup in os.listdir(train_dir)]
	list_newsgroups.sort()
	stemmer=PorterStemmer()
	stop_words=set(stopwords.words('english'))
	def collect_data_from(dir_path,list_newsgroups):
		data=[]
		for group_id,newsgroup in enumerate(list_newsgroups):
			path=dir_path+'\\'+newsgroup+'\\'
			files=[(file_name,path+file_name) for file_name in os.listdir(path)]
			files.sort()
			for file_name,file_path in files:
				with open(file_path,'r') as f:
					text=f.read().lower()
				words=[stemmer.stem(word) for word in re.split('\W+',text) if word not in stop_words and word.isalpha()]
				content=' '.join(words)
				data.append(str(group_id)+'<fff>'+file_name+'<fff>'+content)
		return data
	
	test_data=collect_data_from(test_dir,list_newsgroups)
	train_data=collect_data_from(train_dir,list_newsgroups)
	with open('C:\\Users\\pl\\Downloads\\20news-bydate\\train_processed.txt','w') as f:
		f.write('\n'.join(train_data))
	with open('C:\\Users\\pl\\Downloads\\20news-bydate\\test_processed.txt','w') as f:
		f.write('\n'.join(test_data))

def generate_vocabulary(data_path,min_df=1):
	def compute_idf(df,corpus_size):
		return np.log10(corpus_size/df)
	with open(data_path,'r') as f:
		lines=f.read().splitlines()
	doc_count=defaultdict(int)
	corpus_size=len(lines)
	for line in lines:
		features=line.split('<fff>')
		words=set(features[-1].split())
		for word in words:
			doc_count[word]+=1
	words_idf=[(word,compute_idf(df,corpus_size)) for word,df in doc_count.items() if df>=min_df]
	words_idf.sort(key=lambda tup : -tup[1])
	print("vocabulary size: "+str(len(words_idf)))
	with open('C:\\Users\\pl\\Downloads\\20news-bydate\\words_idf.txt','w')as f:
		f.write('\n'.join([word+'<fff>'+str(idf) for word,idf in words_idf]))

def get_tf_idf(data_path):
	with open('C:\\Users\\pl\\Downloads\\20news-bydate\\words_idf.txt','r')as f:
		words_idf=[(line.split('<fff>')[0], float(line.split('<fff>')[1])) for line in f.read().splitlines()]
	IDF=dict(words_idf)
	ID=dict([(word,index) for index,(word,idf_val) in enumerate(words_idf)])
	with open(data_path,'r') as f:
		docs=[(line.split('<fff>')[0]+'<fff>'+line.split('<fff>')[1]+'<fff>', line.split('<fff>')[2]) for line in f.read().splitlines()]
	data_tf_idf=[]
	for header,text in docs:
		words=[word for word in text.split() if word in IDF]
		word_set=set(words)
		max_tf=max([words.count(word) for word in word_set])
		words_tf_idf=[]
		sum_squares=0.0
		for word in word_set:
			tf=words.count(word)
			tf_idf=(IDF[word]*tf)/max_tf
			words_tf_idf.append((ID[word],tf_idf))
			sum_squares+=tf_idf**2
		words_tf_idf_normalized=[str(index)+':'+str(tf_idf/np.sqrt(sum_squares)) for index,tf_idf in words_tf_idf] 
		data_tf_idf.append(header+' '.join(words_tf_idf_normalized))
	with open('C:\\Users\\pl\\Downloads\\20news-bydate\\tf_idf_vector.txt','w') as f:
		f.write('\n'.join(data_tf_idf))

#gather_data()
generate_vocabulary('C:\\Users\\pl\\Downloads\\20news-bydate\\train_processed.txt',6)
get_tf_idf('C:\\Users\\pl\\Downloads\\20news-bydate\\train_processed.txt')