import numpy as np

def gather_data():
	X,Y=[],[]
	with open('C:\\Users\\pl\\Documents\\x28.txt','r') as f:
		lines=f.read().splitlines()
	for line in lines:
		numbers=line.split()
		row=[]
		for number in numbers[1:16]:
			row.append(float(number))
		X.append(row)
		Y.append(float(numbers[16]))
	return np.array(X),np.array(Y)

def normalize_and_add_1(X):
	X=np.array(X)
	X_max=[]
	for column in range(X.shape[1]):
		Max=0
		for i in X[:,column]:
			if Max<i:
				Max=i
		X_max.append(Max)
	X_max=np.array(X_max)
	X_min=[]
	for column in range(X.shape[1]):
		Min=1e6
		for i in X[:,column]:
			if Min>i:
				Min=i
		X_min.append(Min)
	X_min=np.array(X_min)
	X_normalized=np.array([np.insert((row-X_min)/(X_max-X_min),0,1) for row in X[:]])
	return X_normalized

class RidgeRegression:

	def __init__(seft):
		return

	def fit(seft,X,Y,LAMBDA):
		return np.linalg.inv(X.transpose().dot(X)+LAMBDA*np.identity(X.shape[1])).dot(X.transpose().dot(Y))

	def predict(seft,W,X_new):
		X_new=np.array(X_new)
		return X_new.dot(W)

	def compute_RSS(seft,Y_new,Y_predicted):
		return (1/Y_new.shape[0])*np.sum((Y_new-Y_predicted)**2)

	def get_the_best_LAMBDA(self,X_train,Y_train):
		def cross_validation(num_folds,LAMBDA):
			row_ids=np.array(range(X_train.shape[0]))
			valid_ids=np.split(row_ids[:len(row_ids)-len(row_ids)%num_folds],num_folds)
			valid_ids[-1]=np.append(valid_ids[-1],row_ids[len(row_ids)-len(row_ids)%num_folds:])
			train_ids=[[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
			aver_RSS=0
			for i in range(num_folds):
				valid_part={'X':X_train[valid_ids[i]],'Y':Y_train[valid_ids[i]]}
				train_part={'X':X_train[train_ids[i]],'Y':Y_train[train_ids[i]]}
				W=self.fit(train_part['X'],train_part['Y'],LAMBDA)
				Y_predicted=self.predict(W,valid_part['X'])
				aver_RSS+=self.compute_RSS(valid_part['Y'],Y_predicted)
			return aver_RSS/num_folds

		def range_scan(best_LAMBDA,min_RSS,LAMBDA_values):
			for current_LAMBDA in LAMBDA_values:
				aver_RSS=cross_validation(5,current_LAMBDA)
				if aver_RSS<min_RSS:
					best_LAMBDA=current_LAMBDA
					min_RSS=aver_RSS
			return best_LAMBDA,min_RSS

		best_LAMBDA,min_RSS=range_scan(0,1e8,range(50))
		LAMBDA_values=[k/1e3 for k in range(int(max(0,(best_LAMBDA-1)*1e3,(best_LAMBDA+1)*1e3,1)))]
		best_LAMBDA,min_RSS=range_scan(best_LAMBDA,min_RSS,LAMBDA_values)
		return best_LAMBDA

	def fit_gradient(seft,X_train,Y_train,LAMBDA,learning_rate,max_num_epoch=100,batch_size=128):
		W=np.random.randn(X_train.shape[1])
		last_lose=1e9
		for ep in range(max_num_epoch):
			arr=np.array(range(X_train.shape[0]))
			np.random.shuffle(arr)
			X_train=X_train[arr]
			Y_train=Y_train[arr]
			total_mini_batch=int(np.ceil(X_train.shape[0],batch_size))
			for i in range(total_mini_batch):
				index=i*batch_size
				X_sub=X_train[index:index+batch_size]
				Y_sub=Y_train[index:index+batch_size]
				grad=X_sub.transpose.dot(X_sub.dot(W)-Y_sub)+LAMBDA*W
				W=W-learning_rate*grad
			new_lose=self.compute_RSS(self.predict(W,X_train,Y_train))
			if np.abs(new_lose-last_lose)<=1e-5:
				break
			last_lose=new_lose
		return W
	
X,Y=gather_data()
X=normalize_and_add_1(X)
X_train=X[:50]
Y_train=Y[:50]
X_test=X[50:]
Y_test=Y[50:]
ridge_regression=RidgeRegression()
best_LAMBDA=ridge_regression.get_the_best_LAMBDA(X_train,Y_train)
print('best LAMBDA : ',best_LAMBDA)
W_learned=ridge_regression.fit(X_train,Y_train,best_LAMBDA)
Y_predicted=ridge_regression.predict(W_learned,X_test)
print(ridge_regression.compute_RSS(Y_test,Y_predicted))


