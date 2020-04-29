from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

def load_data(data_path):
	with open(data_path,'r') as f:
		lines=f.read().splitlines();
	label,data=[],[]
	for line in lines:
		feature=line.split('<fff>')
		label.append(int(feature[0]))
		data.append([(int(index_tfidf.split(':')[0]),float(index_tfidf.split(':')[0]))for index_tfidf in feature[2]])
	return data,label

data,label=load_data("C:\\Users\\pl\\Documents\\Python_Project\\ML_DS_2020\\session_1\\TF_IDF\\train_tf_idf_vector.txt")
x=csr_matrix(data)
print(x)

def clustering_with_Kmeans():
	pass
