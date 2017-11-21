from __future__ import absolute_import
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn.decomposition import PCA
import utils.modules as modules
import os
import tempfile

import argparse
import sys

'''
generate the features and store them instead of calculate every time

'''
feature_dim = 43


def gen_feature_file(features_file, data_file, source_file):
	'''
	:param features_file: the extracted feature files
	:param source_file: original data input
	:return: the extracted feature
	'''
	features = []
	data_x = []
	# determine whether we need to generate the files
	if not os.path.exists(features_file):
		# load pos data and convert them to theta and phi
		print 'loading... path=', source_file
		pos_data = modules.load_data(source_file)
		data_x, torsos = modules.get_torso(pos_data)

		for i in range(0, torsos.shape[0]):
			print 'total torso= {0},executing torso:{1}'.format(torsos.shape[0], i)
			u, r, t = modules.get_torso_pca(torsos[i])
			f_angle_3d = modules.export_angle_features_3d(data_x[i], u, r, t)
			f_angle_2d = modules.export_angle_features_2d(data_x[i])
			# f_dis=modules.export_distance_features(data_x[i])
			# flatten the feature data
			f_all = np.concatenate((np.reshape(f_angle_3d, (30,)), np.reshape(f_angle_2d,(13,))), axis=0)
			# f_all = np.reshape(f_angle_3d, (30,))
			features.append(f_all)

		features = np.array(features)
		data_x = np.array(data_x)
		print features.shape
		outfile = open(features_file, 'ab')
		outdata = open(data_file, 'ab')
		np.save(outfile, features)
		outfile.close()
		np.save(outdata, data_x)
		outdata.close()
		print 'finished...'
	else:
		print 'loading features...', features_file
		outfile = open(features_file, 'rb')
		features = np.load(outfile)
		outfile.close()

		print 'loading data...', data_file
		outdata = open(data_file, 'rb')
		data_x = np.load(outdata)
		outdata.close()
		# print features.shape
	return features, data_x

#pass the source folder, distinguish by pos and neg

source_file_pos=["/home/richard/20171117/pos_hug","/home/richard/20171117/pos_cross"]
source_file_neg=["/home/richard/20171117/neg_hug","/home/richard/20171117/neg_bend","/home/richard/20171117/neg_lean","/home/richard/20171117/neg_whatever"]

feature_pos_hug='pos_hug.txt'
feature_neg_hug='neg_hug.txt'
feature_pos_cross='pos_cross.txt'
feature_neg_cross='neg_cross.txt'

data_pos_hug='pos_hug_data.txt'
data_neg_hug='neg_hug_data.txt'
data_pos_cross='pos_cross_data.txt'
data_neg_cross='neg_cross_data.txt'


feature_pos, data_pos = gen_feature_file(feature_pos_hug, data_pos_hug, source_file_pos)
feature_neg, data_neg = gen_feature_file(feature_neg_hug, data_neg_hug, source_file_neg)

# create the y data
y_pos = np.array([[1, 0] for i in range(feature_pos.shape[0])])
y_neg = np.array([[0, 1] for j in range(feature_neg.shape[0])])

data_full_x = np.concatenate((data_pos, data_neg), axis=0)
feature_full_x = np.concatenate((feature_pos, feature_neg), axis=0)
feature_full_x = feature_full_x[:, 0:feature_dim]
feature_full_y = np.concatenate((y_pos, y_neg), axis=0)

# Shuffle data
shuffle_indices = np.random.permutation(np.arange(len(feature_full_x)))
shuffle_x = feature_full_x[shuffle_indices]
shuffle_y = feature_full_y[shuffle_indices]
shuffle_data = data_full_x[shuffle_indices]

testing_size = 100

test_x = shuffle_x[0:testing_size]
test_y = shuffle_y[0:testing_size]

# in order to trace the original inputs
test_x_data = shuffle_data[0:testing_size]

data_x = shuffle_x[testing_size + 1:-1]
data_y = shuffle_y[testing_size + 1:-1]

# parameters
learning_rate = 0.03
# num_steps = 100  #not use any more
batch_size = 30
display_step = 1

print 'total training case number is {0}, total testing case is {1}'.format(len(data_x), len(test_x))
print 'learning rate is {0}, mini batch size is {1}'.format(learning_rate, batch_size)

# Network Parameters
n_hidden_1 = 16  # 1st layer number of neurons
n_hidden_2 = 8 # 2nd layer number of neurons
num_input = feature_dim  # data input
num_classes = 2  # total classes

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias1
weights = {
	'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
	# Hidden fully connected layer with 256 neurons
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	# Hidden fully connected layer with 256 neurons
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	# Output fully connected layer with a neuron for each class
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer


# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# my own mini batch generator
def batch_data(source, target, batch_size):
	# Shuffle data
	shuffle_indices = np.random.permutation(np.arange(len(target)))
	source = source[shuffle_indices]
	target = target[shuffle_indices]

	for batch_i in range(0, len(source) // batch_size):
		start_i = batch_i * batch_size
		source_batch = source[start_i:start_i + batch_size]
		target_batch = target[start_i:start_i + batch_size]
		yield np.array(source_batch), np.array(target_batch)


# Start training
iteration_times=300
# mini batch generator
statistics_accuracy = []
statistics_false_pos=[]
statistics_false_neg=[]
#false positive & false negative
for i in xrange(0, iteration_times):
	batch_generator = batch_data(data_x, data_y, batch_size)
	false_positive = 0
	false_negative = 0
	with tf.Session() as sess:
		# Run the initializer
		sess.run(init)

		for step in range(0, len(data_y) // batch_size):
			batch_x, batch_y = batch_generator.next()
			# Run optimization op (backprop)
			sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
			if step % display_step == 0 or step == 1:
				# Calculate batch loss and accuracy
				loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
				                                                     Y: batch_y})
		# 		print("Step " + str(step) + ", Minibatch Loss= " +
		# 		      "{:.4f}".format(loss) + ", Training Accuracy= " +
		# 		      "{:.3f}".format(acc))
		# print("Optimization Finished!")

		# Calculate accuracy for MNIST test images
		statistics_accuracy.append(100 * sess.run(accuracy, feed_dict={X: test_x,
		                                                               Y: test_y}))

		test_accuracy, test_correct_pred = sess.run([accuracy, correct_pred], feed_dict={X: test_x,
		                                                                                 Y: test_y})

		for p, q in zip(test_correct_pred, test_y):
			if p== False and np.array_equal(q,[1,0]):
				false_positive+=1
			elif p== False and np.array_equal(q,[0,1]):
				false_negative+=1
		statistics_false_pos.append(false_positive/(1.0*testing_size)*100)
		statistics_false_neg.append(false_negative/(1.0*testing_size)*100)

		print("accuracy={0:.2f}%,false positive={1:.2f}%,false negative={2:.2f}%:".format(statistics_accuracy[i],statistics_false_pos[i],statistics_false_neg[i]))

		# outfile = open('test_x_data.txt', 'ab')
		# np.save(outfile, test_x_data)
		# outfile.close()
		#
		# outfile = open('test_x.txt', 'ab')
		# np.save(outfile, test_x)
		# outfile.close()
		#
		# outfile = open('test_y.txt', 'ab')
		# np.save(outfile, test_y)
		# outfile.close()
		#
		# outfile = open('test_correct_pred.txt', 'ab')
		# np.save(outfile, test_correct_pred)
		# outfile.close()
#
#
accuracy_file = 'statistics_accuracy_' + str(feature_dim) + '.txt'
outfile = open(accuracy_file, 'wb')
np.save(outfile, statistics_accuracy)
outfile.close()
print 'accuracy average=', np.average(statistics_accuracy)

false_positive_file = 'statistics_false_pos_' + str(feature_dim) + '.txt'
outfile = open(false_positive_file, 'wb')
np.save(outfile, statistics_false_pos)
outfile.close()
print 'false positive average=', np.average(statistics_false_pos)

false_negative_file = 'statistics_false_neg_' + str(feature_dim) + '.txt'
outfile = open(false_negative_file, 'wb')
np.save(outfile, statistics_false_neg)
outfile.close()
print 'false negative average=', np.average(statistics_false_neg)
