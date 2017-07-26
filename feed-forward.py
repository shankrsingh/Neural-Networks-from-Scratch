import numpy as np
import matplotlib.pyplot as plt

Nclass=500
X1=np.random.randn(Nclass,2)+np.array([0,-2])
X2=np.random.randn(Nclass,2)+np.array([2,2])
X3=np.random.randn(Nclass,2)+np.array([-2,2])
X=np.vstack([X1,X2,X3])
Y=np.array([0]*Nclass+[1]*Nclass + [2]*Nclass)

d=2  #Features
m=3  #hidden
k=4 #op

w1=np.random.randn(d,m)  # weight1 =  inner dimensions * size

b1=np.random.randn(m)  #bias1 = size of hidden layer

#v
w2=np.random.randn(m,k) #weight2 = inner dimension * no. of outputs

b2=np.random.randn(k) #bias2 = size of outputs

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def softmax(A):
	expA=np.exp(A)
	Y=expA/ expA.sum(axis=1,keepdims=True)
	return Y

def forward_prop(X,w1,b1,w2,b2):
	Z=X.dot(w1)+b1
	A=sigmoid(Z)
	Y=softmax(A)
	return Y

def classification_rate(orig,predicted):
	total=0
	true=0

	for i in range(len(orig)):
		total+=1
		if(orig[i]==P[i]):
			true+=1


	return float(true/total)
Y_given_X=forward_prop(X,w1,b1,w2,b2)
P=np.argmax(Y_given_X,axis=1)  # gives the index of max element row-wise

print("Classification Rate:",classification_rate(Y,P))
