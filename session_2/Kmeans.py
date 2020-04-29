from collections import defaultdict
import numpy as np

class Member:
	def __init__(self,rd,label=None,doc_id=None):
		self.rd=rd
		self.label=label
		self.doc_id=doc_id

class Cluster:
	def __init__(self):
		self.centroid = None
		self.members = []

	def reset_member(self):
		self.members = []

	def add_member(self,member):
		self.members.append(member)

class Kmeans:
	def __init__(self, num_clusters):
		self.num_clusters = num_clusters
		self.clusters = [Cluster() for i in range(self.num_clusters)]
		self.E=[]
		self.S=0

	#doc du lieu tu file va chuyen ve dang ma tran thua
	def load_data(self,data_path):
		def sparse_to_dense(sparse_rd,vocab_size):
			rd=[0 for i in range(vocab_size)]
			indices_tfidfs=sparse_rd.split()
			for index_tfidf in indices_tfidfs:
				index=int(index_tfidf.split(':')[0])
				tfidf=float(index_tfidf.split(':')[1])
				rd[index]=tfidf
			return np.array(rd)

		with open(data_path,'r') as f:
			lines=f.read().splitlines()
		with open("C:\\Users\\pl\\Documents\\Python_Project\\ML_DS_2020\\session_1\\TF_IDF\\words_idf.txt",'r') as f:
			self.vocab_size=len(f.read().splitlines())

		self.data=[]
		self.label_count=defaultdict(int)
		for line in lines:
			feature=line.split('<fff>')
			label,doc_id=int(feature[0]),int(feature[1])
			self.label_count[label]+=1
			rd=sparse_to_dense(feature[2],self.vocab_size)
			self.data.append(Member(rd,label,doc_id))

	#khoi tao tam cum
	def random_init(self,seed_value):
		for cluster in self.clusters:
			cluster.centroid=self.data[np.random.randint(0,len(self.data))].rd

	#tinh khoang cach
	def compute_distance(self,member,centroid):
		return np.sqrt(np.sum((member.rd-centroid)**2))

	#chon cum cho member
	def select_cluser_for(self,member):
		best_fit_cluster=None
		min_distance=1e9
		for cluster in self.clusters:
			distance=self.compute_distance(member,cluster.centroid)
			if distance<min_distance:
				min_distance=distance
				best_fit_cluster=cluster
		best_fit_cluster.add_member(member)
		return min_distance

	#cap nhap tam cum
	def update_centroid_of(self,cluster):
		member_rds=[member.rd for member in cluster.members]
		aver_rd=np.mean(member_rds,axis=0)
		sqrt_sum_sqr=np.sqrt(np.sum(aver_rd**2))
		new_centroid=np.array([value/sqrt_sum_sqr for value in aver_rd])
		cluster.centroid=new_centroid

	#dieu kien dung
	def stopping_condition(self,criterion,threshold):
		criteria=['centroid','similarity','max_iters']
		assert criterion in criteria
		if criterion=='max_iters':
			if self.iteration>=threshold:
				return True
			else:
				return False
		elif criterion=='centroid':
			E_new=[list(cluster.centroid)for cluster in self.clusters]
			E_new_minus_E=[centroid for centroid in E_new if centroid not in self.E]
			self.E=E_new
			if len(E_new_minus_E)<=threshold:
				return True
			else:
				return False
		else:
			new_S_minus_S=self.new_S-self.S
			self.S=self.new_S
			if new_S_minus_S<=threshold:
				return True
			else:
				return False
	
	#chuay thuat toan Kmeans		
	def run(self,seed_value,criterion,threshold):
		self.random_init(seed_value)
		self.iteration=0
		while True:
			for cluster in self.clusters:
				cluster.reset_member()
			self.new_S=0
			for member in self.data:
				max_S=self.select_cluser_for(member)
				self.new_S+=max_S
			for cluser in self.clusters:
				self.update_centroid_of(cluster)
			self.iteration+=1
			if self.stopping_condition(criterion,threshold):
				break

	#danh gia hieu qua phan cum
	def compute_purity(self):
		majority_sum=0
		for cluster in self.clusters:
			member_labels=[member.label for member in cluster.members]
			max_count=max([member_labels.count(label) for label in range(20)])
			majority_sum+=max_count
		return majority_sum/len(self.data)

	def compute_NMI(self):
		I_value,H_omega,H_C,N=0,0,0,len(self.data)
		for cluster in self.clusters:
			wk=len(cluster.members)
			H_omega+=-(wk/N)*np.log10(wk/N)
			member_labels=[member.label for member in cluster.members]
			for label in range(20):
				wk_cj=member_labels.count(label)
				cj=self.label_count[label]
				I_value+=wk_cj/N*np.log10(N*wk_cj/(wk*cj)+1e-12)
		for label in range(20):
			cj=self.label_count[label]
			H_C+=-cj/N*np.log10(cj/N)
		return I_value*2/(H_omega+H_C)

km=Kmeans(20)
km.load_data("C:\\Users\\pl\\Documents\\Python_Project\\ML_DS_2020\\session_1\\TF_IDF\\train_tf_idf_vector.txt")
km.run(0,'centroid',2)
print(km.compute_purity())
print(km.compute_NMI())