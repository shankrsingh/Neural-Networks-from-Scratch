import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np


#4-layer sizes
numHidden=4
h= [500,600,500,600]
x_size=784
output_size=10
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

"""
w0=tf.Variable(tf.random_normal(dtype=tf.float32,[x_size,h[0]]))
b0=tf.Variable(tf.random_normal([h[0]]))
w0=tf.Variable(tf.random_normal(dtype=tf.float32,[x_size,h[0]]))
b0=tf.Variable(tf.random_normal([h[0]]))
"""
layers=[]
dictx={}
def neuralnet(x,numHidden,h):
	for i in range(numHidden):
		weight='w'+str(i)
		bias='b'+str(i)
		if(i==0):
			dictx[weight]=tf.Variable(tf.random_normal([x_size,h[i]]))
			dictx[bias]=tf.Variable(tf.random_normal([h[i]]))
		elif(i==numHidden-1):
			dictx[weight]=tf.Variable(tf.random_normal([h[i-1],output_size]))
			dictx[bias]=tf.Variable(tf.random_normal([output_size]))

		else:
			dictx[weight]=tf.Variable(tf.random_normal([h[i-1],h[i]]))
			dictx[bias]=tf.Variable(tf.random_normal([h[i]]))	    	
	    	
	     	# w1 = h0 * h1  ,, b1= h1
	for i in range(numHidden):
			weight='w'+str(i)
			bias='b'+str(i)
			#layer=str(i)
			
			if(i==0):
				layers.append(tf.nn.relu(tf.matmul(x,dictx[weight])+dictx[bias]) )
			elif(i==numHidden-1):
				outputlayer=tf.matmul(layers[i-1],dictx[weight])+dictx[bias]
			else:
				#l=int(i)-1
				layers.append(tf.nn.relu(tf.matmul(layers[i-1],dictx[weight])+dictx[bias]  ))

	return outputlayer


def train(x):
	prediction=neuralnet(x)
	cost=tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits(prediction,y) )    # t.logy   t=prediction
	optimizer=tf.train.AdamOptimizer().minimize(cost)
	epox=10
	with tf.Session() as sess:
		for epochs in range(epox):
			epoch_loss=0
			

	





"""
i=0

weight=w0
bias=b0
layer=0

layers[0]=x * dictx[w0] + dictx[b0]

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

i=1

weight=w1
bias=b1
layer=1
l=1-1=0
layers[1]=layers[0] * dictx[w1] + dictx[b1]
 
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

i=2

weight=w2
bias = b1
layer=2

l=2-1=1

layers[2]=layers[1] * dictx[w2]+dictx[b2]

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx

"""




if __name__ == '__main__':
	neuralnet(x,numHidden,h)
	    






