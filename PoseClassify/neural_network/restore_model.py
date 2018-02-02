import tensorflow as tf
import numpy as np
import utils.modules as modules

def neural_network_classify(x,para):
	# extract the features

	# step1: get torso 7 points
	torsoIndices = [8, 20, 4, 1, 16, 0, 12]
	torsoMat = []
	for i in torsoIndices:
		torsoMat.append(x[i])

	#step2: export the pca dimension
	u, r, t = modules.get_torso_pca(torsoMat)

	#step3: generate features
	f_angle_3d = modules.export_angle_features_3d(x, u, r, t)
	f_angle_2d = modules.export_angle_features_2d(x)
	f_combine = np.concatenate((np.reshape(f_angle_3d, (33,)), np.reshape(f_angle_2d,(18,))), axis=0)

	#step4: determine the output class
	layer_1 = np.add(np.matmul(f_combine, para[0]), para[3])
	layer_2 = np.add(np.matmul(layer_1, para[1]), para[4])
	out_layer = np.matmul(layer_2, para[2]) + para[5]
	return f_combine,out_layer

def restore_parameters():
	# Network Parameters
	n_hidden_1 = 12  # 1st layer number of neurons
	n_hidden_2 = 6 # 2nd layer number of neurons
	num_input = feature_dim=51  # data input
	num_classes = 2  # total classes

	# Create some variables.
	h1=tf.get_variable("h1", [num_input, n_hidden_1])
	h2=tf.get_variable("h2", [n_hidden_1, n_hidden_2])
	h3=tf.get_variable("h3", [n_hidden_2, num_classes])

	b1=tf.get_variable("b1",[n_hidden_1])
	b2=tf.get_variable("b2",[n_hidden_2])
	b3=tf.get_variable("b3",[num_classes])

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()

	# Later, launch the model, use the saver to restore variables from disk, and
	# do some work with the model.
	sess=tf.Session()

	saver = tf.train.import_meta_graph('.\\pose_model.ckpt.meta')
	saver.restore(sess,tf.train.latest_checkpoint('.\\'))

	#saver.restore(sess, ".\\pose_model.ckpt")
	print("Model restored.")
	print(sess.run('h1:0'))

	#print("b1 : %s" % sess.run(b1))
	para=[]
	para.append(sess.run('h1:0'))
	para.append(sess.run('h2:0'))
	para.append(sess.run('h3:0'))
	para.append(sess.run('b1:0'))
	para.append(sess.run('b2:0'))
	para.append(sess.run('b3:0'))
	return para
