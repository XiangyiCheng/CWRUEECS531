import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('/tmp/data/',one_hot=True)


mnist_data=tf.placeholder('float',[None,784])
label=tf.placeholder('float')

# 10 different digit numbers: 0-9
n_classes=10
n_each_group=100


def baseline_neural_network(data):
	# input_data * weight + bias
	# the 
	n_hidden1=500

	# each image has 784 pixels which is calculated by 28*28.
	hidden_layer1={'weights':tf.Variable(tf.random_normal([784,n_hidden1])),'biases':tf.Variable(tf.random_normal([n_hidden1]))}

	output_layer={'weights':tf.Variable(tf.random_normal([n_hidden1,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

	layer1 = tf.add(tf.matmul(data,hidden_layer1['weights']),hidden_layer1['biases'])
	act_layer1 = tf.nn.relu(layer1)

	model = tf.matmul(act_layer1,output_layer['weights'])+output_layer['biases']

	return model 



def deep_neural_network(data):
	# input_data * weight + bias
	n_hidden1=500
	n_hidden2=1000
	n_hidden3=1500
	n_hidden4=2000
	n_hidden5=2500
	n_hidden6=2500
	n_hidden7=2500
	n_hidden8=2500

	hidden_layer1={'weights':tf.Variable(tf.random_normal([784,n_hidden1])),'biases':tf.Variable(tf.random_normal([n_hidden1]))}
	hidden_layer2={'weights':tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),'biases':tf.Variable(tf.random_normal([n_hidden2]))}
	hidden_layer3={'weights':tf.Variable(tf.random_normal([n_hidden2,n_hidden3])),'biases':tf.Variable(tf.random_normal([n_hidden3]))}
	hidden_layer4={'weights':tf.Variable(tf.random_normal([n_hidden3,n_hidden4])),'biases':tf.Variable(tf.random_normal([n_hidden4]))}
	hidden_layer5={'weights':tf.Variable(tf.random_normal([n_hidden4,n_hidden5])),'biases':tf.Variable(tf.random_normal([n_hidden5]))}
	hidden_layer6={'weights':tf.Variable(tf.random_normal([n_hidden5,n_hidden6])),'biases':tf.Variable(tf.random_normal([n_hidden6]))}
	hidden_layer7={'weights':tf.Variable(tf.random_normal([n_hidden6,n_hidden7])),'biases':tf.Variable(tf.random_normal([n_hidden7]))}
	hidden_layer8={'weights':tf.Variable(tf.random_normal([n_hidden7,n_hidden8])),'biases':tf.Variable(tf.random_normal([n_hidden8]))}


	output_layer1={'weights':tf.Variable(tf.random_normal([n_hidden1,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

	output_layer2={'weights':tf.Variable(tf.random_normal([n_hidden2,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

	output_layer8={'weights':tf.Variable(tf.random_normal([n_hidden8,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}


	layer1 = tf.add(tf.matmul(data,hidden_layer1['weights']),hidden_layer1['biases'])
	act_layer1 = tf.nn.relu(layer1)

	layer2 = tf.add(tf.matmul(act_layer1,hidden_layer2['weights']),hidden_layer2['biases'])
	act_layer2 = tf.nn.relu(layer2)	

	layer3 = tf.add(tf.matmul(act_layer2,hidden_layer3['weights']),hidden_layer3['biases'])
	act_layer3 = tf.nn.relu(layer3)

	layer4 = tf.add(tf.matmul(act_layer3,hidden_layer4['weights']),hidden_layer4['biases'])
	act_layer4 = tf.nn.relu(layer4)

	layer5 = tf.add(tf.matmul(act_layer4,hidden_layer5['weights']),hidden_layer5['biases'])
	act_layer5 = tf.nn.relu(layer5)

	layer6 = tf.add(tf.matmul(act_layer5,hidden_layer6['weights']),hidden_layer6['biases'])
	act_layer6 = tf.nn.relu(layer6)

	layer7 = tf.add(tf.matmul(act_layer6,hidden_layer7['weights']),hidden_layer7['biases'])
	act_layer7 = tf.nn.relu(layer7)

	layer8 = tf.add(tf.matmul(act_layer7,hidden_layer8['weights']),hidden_layer8['biases'])
	act_layer8 = tf.nn.relu(layer8)


	model_1 = tf.matmul(act_layer1,output_layer1['weights'])+output_layer1['biases']

	model_2 = tf.matmul(act_layer2,output_layer2['weights'])+output_layer2['biases']

	model_8 = tf.matmul(act_layer8,output_layer8['weights'])+output_layer8['biases']


	return model_1,model_2,model_8



def convolutional_neural_network(data):

	weights={'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
	         'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),'out':tf.Variable(tf.random_normal([1024,n_classes]))}
	biases={'b_conv1':tf.Variable(tf.random_normal([32])),'b_conv2':tf.Variable(tf.random_normal([64])),
	         'b_fc':tf.Variable(tf.random_normal([1024])),'out':tf.Variable(tf.random_normal([n_classes]))}

	data=tf.reshape(data,shape=[-1,28,28,1])

	conv1=tf.nn.conv2d(data,weights['W_conv1'],strides=[1,1,1,1],padding='SAME')
	conv1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	conv2=tf.nn.conv2d(conv1,weights['W_conv2'],strides=[1,1,1,1],padding='SAME')
	conv2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	fc=tf.reshape(conv2,[-1,7*7*64])
	fc=tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])

	model_CNN=tf.matmul(fc,weights['out'])+biases['out']

	return model_CNN 




def train_CNN(models):

	prep_model=convolutional_neural_network(models)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prep_model, labels=label))
	opitimizer=tf.train.AdamOptimizer().minimize(cost)

	step_epochs=10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(step_epochs):
			Loss=0
			for _ in range(int(mnist.train.num_examples/n_each_group)):
				epoch_x,epoch_y = mnist.train.next_batch(n_each_group)
				_,loss = sess.run([opitimizer,cost],feed_dict={models:epoch_x,label:epoch_y})
				Loss+=loss
			print 'Step',epoch,'is completed out of',step_epochs, '. Loss is ',Loss

		correct=tf.equal(tf.argmax(prep_model,1),tf.argmax(label,1))
		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		print 'CNN accuracy:',accuracy.eval({models:mnist.test.images,label:mnist.test.labels})






def train_neural_network(models):

	prep_model1,prep_model2,prep_model8=deep_neural_network(models)

	cost1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prep_model1, labels=label))
	opitimizer1=tf.train.AdamOptimizer().minimize(cost1)

	cost2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prep_model2, labels=label))
	opitimizer2=tf.train.AdamOptimizer().minimize(cost2)

	cost8=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prep_model8, labels=label))
	opitimizer8=tf.train.AdamOptimizer().minimize(cost8)

	step_epochs=10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# baseline
		for epoch in range(step_epochs):
			Loss=0
			for _ in range(int(mnist.train.num_examples/n_each_group)):
				epoch_x,epoch_y = mnist.train.next_batch(n_each_group)
				_,loss = sess.run([opitimizer1,cost1],feed_dict={models:epoch_x,label:epoch_y})
				Loss+=loss

		correct1=tf.equal(tf.argmax(prep_model1,1),tf.argmax(label,1))
		accuracy1=tf.reduce_mean(tf.cast(correct1,'float'))
		print 'baseline accuracy:',accuracy1.eval({models:mnist.test.images,label:mnist.test.labels})


		# 2 layes
		for epoch in range(step_epochs):
			Loss=0
			for _ in range(int(mnist.train.num_examples/n_each_group)):
				epoch_x,epoch_y = mnist.train.next_batch(n_each_group)
				_,loss = sess.run([opitimizer2,cost2],feed_dict={models:epoch_x,label:epoch_y})
				Loss+=loss

		correct2=tf.equal(tf.argmax(prep_model2,1),tf.argmax(label,1))
		accuracy2=tf.reduce_mean(tf.cast(correct2,'float'))
		print '2 layers accuracy:',accuracy2.eval({models:mnist.test.images,label:mnist.test.labels})


		# 8 layers
		for epoch in range(step_epochs):
			Loss=0
			for _ in range(int(mnist.train.num_examples/n_each_group)):
				epoch_x,epoch_y = mnist.train.next_batch(n_each_group)
				_,loss = sess.run([opitimizer8,cost8],feed_dict={models:epoch_x,label:epoch_y})
				Loss+=loss
			print 'Step',epoch,'is completed out of',step_epochs,'. Loss is ',Loss

		correct8=tf.equal(tf.argmax(prep_model8,1),tf.argmax(label,1))
		accuracy8=tf.reduce_mean(tf.cast(correct8,'float'))
		print '8 layers accuracy:',accuracy8.eval({models:mnist.test.images,label:mnist.test.labels})



def train_baseline(models):

	prep_model=baseline_neural_network(models)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prep_model, labels=label))
	opitimizer=tf.train.AdamOptimizer().minimize(cost)

	step_epochs=10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(step_epochs):
			Loss=0
			for _ in range(int(mnist.train.num_examples/n_each_group)):
				epoch_x,epoch_y = mnist.train.next_batch(n_each_group)
				_,loss = sess.run([opitimizer,cost],feed_dict={models:epoch_x,label:epoch_y})
				Loss+=loss
			print 'Step',epoch,'is completed out of',step_epochs, '. Loss is ',Loss

		correct=tf.equal(tf.argmax(prep_model,1),tf.argmax(label,1))
		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		print 'baseline accuracy:',accuracy.eval({models:mnist.test.images,label:mnist.test.labels})

train_neural_network(mnist_data)
#train_baseline(mnist_data)
#train_CNN(mnist_data)