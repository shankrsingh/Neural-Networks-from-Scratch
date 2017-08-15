import numpy as np


def softmax(z):
	expa=np.exp(z)
	return expa/expa.sum(axis=1,keepdims=True)

def sigmoid(a):
	return 1/1+(np.exp(-a))

def forward(x,y,w1,b1,w2,b2):
	z=sigmoid(x.dot(w1)+b1)
	y=softmax(z.dot(w2)+b2)

	return z,y

def cross_entropy(t,y):
	return np.mean(t* np.log(y))

def error_rate(t,y):
	return np.mean(np.argmax(y)!=t)


def dw2(t,y,z):
	return z.T.dot(t-y)

def db2(t,y):
	return (t-y).sum(axis=0)

def dw1(x,y,t,w2,z):
	dz=(t-y).dot(w2.T)*z*(1-z)
	return x.T.dot(dz)

def db1(y,t,z,w2):
	return ((t-y).dot(w2.T)*z*(1-z)).sum(axis=0)


def main():
	for epochs in range(1000):
		z,y=forward(x,y,w1,b1,w2,b2)
		
		if epoch%100==0:
			costs=cross_entropy(t,y)
			error=error_rate(t,y)

			print('Error at epoch',epochs,'is',error,' Cost is ',costs)

		#Gradient Updates.

		w2+=learning_rate*dw2(t,y,z)
		b2+=learning_rate*db2(t,y)
		w1+=learning_rate*dw1(x,y,t,w2,z)
		b1+=learning_rate*db1(y,t,z,w2)


if __name__ == '__main__':
    main()
