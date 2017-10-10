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
def gen_feature_file(features_file, source_file):
    features = []
    # determine whether we need to generate the files
    if not os.path.isfile(features_file):
        outfile = open(features_file, 'ab')
        # load pos data and convert them to theta and phi
        for p in source_file:
            print 'loading... path=', p
            pos_data = modules.load_data(p)
            data_x, torsos = modules.get_torso(pos_data)

            for i in range(0, torsos.shape[0]):
                print 'total torso= {0},executing torso:{1}'.format(torsos.shape[0], i)
                u, r, t = modules.get_torso_pca(torsos[i])
                f = modules.export_features(data_x[i], u, r, t)
                #flatten the feature data
                features.append(np.reshape(f,(18,)))

        features = np.array(features)
        print features.shape
        np.save(outfile, features)
        outfile.close()
        print 'finished...'
    else:
        print 'loading features...',features_file
        outfile = open(features_file, 'rb')
        features = np.load(outfile)
        outfile.close()
        #print features.shape
    return features

feature_pos_file='f_pos.txt'
data_pos_files=['./kinect_data/jointPos_Richard_Pos.txt','./kinect_data/jointPos_Dexter_Pos.txt','./kinect_data/jointPos_Jay_Pos.txt']
#data_pos_files=['./kinect_data/jointPos_Richard_Pos.txt']
feature_neg_file='f_neg.txt'
data_neg_files=['./kinect_data/jointPos_Richard_Neg.txt','./kinect_data/jointPos_Dexter_Neg.txt','./kinect_data/jointPos_Jay_Neg.txt']
#data_neg_files=['./kinect_data/jointPos_Richard_Neg.txt']

data_pos=gen_feature_file(feature_pos_file,data_pos_files)
data_neg=gen_feature_file(feature_neg_file,data_neg_files)

#create the y data
y_pos=np.array([[1,0] for i in range(data_pos.shape[0])])
y_neg=np.array([[0,1] for j in range(data_neg.shape[0])])

full_x=np.concatenate((data_pos,data_neg),axis=0)
full_y=np.concatenate((y_pos,y_neg),axis=0)

# Shuffle data
shuffle_indices = np.random.permutation(np.arange(len(full_x)))
full_x = full_x[shuffle_indices]
full_y = full_y[shuffle_indices]

test_x=full_x[0:200]
test_y=full_y[0:200]

data_x=full_x[201:-1]
data_y=full_y[201:-1]



# parameters
learning_rate = 0.05
num_steps = 100  #not use any more
batch_size = 50
display_step = 1

print 'total training case number is {0}, total testing case is {1}'.format(len(data_x),len(test_x))
print 'learning rate is {0}, mini batch size is {1}'.format(learning_rate,batch_size)

# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons
num_input = 18 # MNIST data input (img shape: 28*28)
num_classes = 2 # MNIST total classes (0-9 digits)

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
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

   for batch_i in range(0, len(source)//batch_size):
      start_i = batch_i * batch_size
      source_batch = source[start_i:start_i + batch_size]
      target_batch = target[start_i:start_i + batch_size]
      yield np.array(source_batch), np.array(target_batch)

# Start training

#mini batch generator
batch_generator = batch_data(data_x,data_y,batch_size)

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(0,len(data_y)//batch_size):
        batch_x, batch_y = batch_generator.next()
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:",
        sess.run(accuracy, feed_dict={X:test_x,
                                      Y: test_y}))
