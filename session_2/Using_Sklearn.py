from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np

#chuyen du lieu thanh ma tran vector va ma tran nhan lop
def load_data(data_path):
	with open(data_path,'r') as f:
		lines=f.read().splitlines();
	labels,tfidf_values,row,col=[],[],[],[]
	for line_id,line in enumerate(lines):
		feature=line.split('<fff>')
		labels.append(int(feature[0]))
		for index_tfidf in feature[2].split():
			tfidf_values.append(float(index_tfidf.split(':')[1]))
			col.append(int(index_tfidf.split(':')[0]))
			row.append(line_id)
	#Transform to sparse matrix
	data=csr_matrix((tfidf_values,(row,col)))
	return data,np.array(labels)

#tinh do chinh xac cua ket qua phan lop
def compute_accuracy(predicted,expected):
	matches=np.equal(predicted,expected)
	accuracy=np.sum(matches)/len(expected)
	return accuracy

#danh gia chat luong phan cum
def compute_purity(predicted,expected):
  majority_sum=0
  for cluster_index in range(20):
    member_indexs=np.where(predicted==cluster_index)[0]
    expected_labels=[expected[index]for index in member_indexs]
    max_count=max(expected_labels.count(label)for label in range(20))
    majority_sum+=max_count
  print(majority_sum)
  return majority_sum/len(expected)

def clustering_with_Kmeans(train_data,train_labels,test_data,test_labels):
	kmeans=KMeans(
		n_clusters=20,
		init='random',
		n_init=5,tol=1e-3,
		random_state=2020)
	kmeans.fit(train_data)
	test_predicted=kmeans.predict(test_data)
	print('train purity:',compute_purity(kmeans.labels_,train_labels))
	print('test purity:',compute_purity(test_predicted,test_labels))

def classifying_with_linear_SVMs(train_data,train_labels,test_data,test_labels):
	classifier=LinearSVC(
		C=10,
		tol=1e-3,
		verbose=True)
	classifier.fit(train_data,train_labels)
	predicted=classifier.predict(test_data)
	accuracy=compute_accuracy(predicted=predicted,expected=test_labels)
	print('linear SVMs accuracy:',accuracy)
 
def classifying_with_kernel_SVMs(train_data,train_labels,test_data,test_labels):
	classifier=SVC(
		C=50,
		kernel='rbf',
		gamma=0.1,
		tol=1e-3,
		verbose=True)
	classifier.fit(train_data,train_labels)
	predicted=classifier.predict(test_data)
	accuracy=compute_accuracy(predicted=predicted,expected=test_labels)
	print('kernel SVMs accuracy:',accuracy)

#RUN

train_data,train_labels=load_data(
	"C:\\Users\\pl\\Documents\\Python_Project\\ML_DS_2020\\session_1\\TF_IDF\\train_tf_idf_vector.txt")
test_data,test_labels=load_data(
	"C:\\Users\\pl\\Documents\\Python_Project\\ML_DS_2020\\session_1\\TF_IDF\\test_tf_idf_vector.txt")
clustering_with_Kmeans(train_data,train_labels,test_data,test_labels)
classifying_with_linear_SVMs(train_data,train_labels,test_data,test_labels)
classifying_with_kernel_SVMs(train_data,train_labels,test_data,test_labels)