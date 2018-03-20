import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('/tmp/data/',one_hot=True)



n_classes=10
batch_size=100

# x is the data. y is the label of the data.
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')


def baseline_neural_network(data):
	# input_data * weight + bias
	n_nodes_hl1=500

	hidden_1_layer={'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	output = tf.matmul(l1,output_layer['weights'])+output_layer['biases']

	return output 



def neural_network_model(data):
	# input_data * weight + bias
	n_nodes_hl1=500
	n_nodes_hl2=1000
	n_nodes_hl3=1500
	n_nodes_hl4=2000
	n_nodes_hl5=2500

	hidden_1_layer={'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	hidden_4_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4])),'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
	hidden_5_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl4,n_nodes_hl5])),'biases':tf.Variable(tf.random_normal([n_nodes_hl5]))}
	output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl5,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)	

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']),hidden_4_layer['biases'])
	l4 = tf.nn.relu(l4)

	l5 = tf.add(tf.matmul(l4,hidden_5_layer['weights']),hidden_5_layer['biases'])
	l5 = tf.nn.relu(l5)

	output = tf.matmul(l5,output_layer['weights'])+output_layer['biases']

	return output 



def convolutional_neural_network(x):

	weights={'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
	         'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),'out':tf.Variable(tf.random_normal([1024,n_classes]))}
	biases={'b_conv1':tf.Variable(tf.random_normal([32])),'b_conv2':tf.Variable(tf.random_normal([64])),
	         'b_fc':tf.Variable(tf.random_normal([1024])),'out':tf.Variable(tf.random_normal([n_classes]))}

	x=tf.reshape(x,shape=[-1,28,28,1])

	conv1=tf.nn.conv2d(x,weights['W_conv1'],strides=[1,1,1,1],padding='SAME')
	conv1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	conv2=tf.nn.conv2d(conv1,weights['W_conv2'],strides=[1,1,1,1],padding='SAME')
	conv2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	fc=tf.reshape(conv2,[-1,7*7*64])
	fc=tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])

	output=tf.matmul(fc,weights['out'])+biases['out']

	return output 




def train_neural_network(x):
	prediction=neural_network_model(x)
	#prediction=convolutional_neural_network(x)
	#prediction=baseline_neural_network(x)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	opitimizer=tf.train.AdamOptimizer().minimize(cost)

	hm_epochs=10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss=0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				_,c = sess.run([opitimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
				epoch_loss+=c
			print 'Epoch',epoch,'completed out of',hm_epochs,'loss',epoch_loss


		correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		print 'accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels})

train_neural_network(x)
