import numpy as np
import tensorflow as tf
import utils.extmath
import os
import time
import matplotlib.pyplot as plt

def timeit(method):
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()
		if 'log_time' in kw:
			name = kw.get('log_name', method.__name__.upper())
			kw['log_time'][name] = int((te - ts) * 1000)
		else:
			print '%r  %2.2f ms' % \
			      (method.__name__, (te - ts) * 1000)
		return result

	return timed


def load_data(fpath):
	'''
	:param fname: load the data at 'fpath'
	:return:
	'''
	data = []
	l=len(fpath)
	for p in fpath:
		for dirpath, subdirs, files in os.walk(p):
			for x in files:
				if x.endswith('.txt'):
					with open(os.path.join(dirpath, x), 'r') as f:
						for line in f.readlines():
							data.append(line.replace('\n', '').split(' '))
						f.close()
	return data


def get_torso(data):
	'''
	:param data: n sets of skeleton data (25 points per set)
	:return: 25 points data set. torso matrix
	'''
	dataNum = len(data) / 25
	dataX = []

	for j in xrange(0, dataNum):
		dataSet = np.zeros((25, 3))
		for i in xrange(0, 25):
			# data[0][1][2:-1]
			dataSet[i] = np.array(
				[float(data[i + j * 25][1][2:-1]), float(data[i + j * 25][2][2:-1]), float(data[i + j * 25][3][2:-1])])
		dataX.append(dataSet)
	dataX = np.array(dataX)
	# dataY = np.array([[1, 0] for x in range(len(dataX))])

	# get torso 7 points
	torsoIndices = [8, 20, 4, 1, 16, 0, 12]
	torsoMat = []
	torsoSet = []
	for j in range(len(dataX)):
		for i in torsoIndices:
			torsoMat.append(dataX[j][i])
		torsoSet.append(torsoMat)
		torsoMat = []
	torsoSet = np.array(torsoSet)
	# print 'torsoMatrix shape=', torsoSet.shape
	return dataX, torsoSet



# @timeit
def get_torso_pca(torsoMat, factors=2):
	# tensorflow
	config = tf.ConfigProto()
	config.intra_op_parallelism_threads = 44
	config.inter_op_parallelism_threads = 44
	with tf.Session(config=config) as sess:
		# SVD
		# Center the points
		torsoPca = torsoMat - np.mean(torsoMat, axis=0)
		# torsoPca=np.matmul(torsoPca,np.transpose(torsoPca))
		St, Ut, Vt = tf.svd(torsoPca, full_matrices=False)
		# print 'Ut=\n', sess.run(Ut)
		# print 'St=\n', sess.run(St)
		# print 'Vt=\n', sess.run(Vt)

		# Compute reduced matrices
		Sk = tf.diag(St)[0:factors, 0:factors]
		Vk = Vt[:, 0:factors]
		Uk = Ut[0:factors, :]

		# print 'Vk=\n', sess.run(Vk)

		# Compute user average rating
		# torsoNew = sess.run(tf.matmul(torsoPca, Vk))
		# print 'torso new.shape', torsoNew.shape

		u = sess.run(Vk)[:, 0]
		r = sess.run(Vk)[:, 1]

		# u top-down
		max_abs_cols = np.argmax(np.abs(u))
		signs = np.sign(u[max_abs_cols])
		u *= -signs
		# r left right

		# t = u X r
		t = np.cross(u, r)
		sess.close()
	# print 'u=', u
	# print 'r=', r
	# print 't=', t
	tf.reset_default_graph()
	return u, r, t

def export_distance_features(pose_set):
	center_p=pose_set[1]
	for i in pose_set:
		utils.extmath.distanse_3d(i[0],i[1],i[2],center_p[0],center_p[1],center_p[2])
	pass

def get_vector_angles(v,u,r):
	vu_theta=utils.extmath.vector_angle(v,u)
	vr_theta=utils.extmath.vector_angle(v,r)
	if vu_theta<90:
		vr_theta=360-vr_theta
	return vr_theta

def export_angle_features_2d(pose_set):
	'''
	using pca to project the 25 joint map on a plane, calculate the angles between pca u and joint vectors.
	:param pose_set: 25 joint map
	:return: angle features in 2d plane
	'''
	u,r,t=get_torso_pca(pose_set)
	pose_set_2d=[]
	for x in pose_set:
		pose_set_2d.append([np.matmul(x,u),np.matmul(x,r)])
	pose_set_2d=np.array(pose_set_2d)
	# treat [1] as center
	center=pose_set_2d[1]
	pose_set_2d_centered=[]
	for x in pose_set_2d:
		pose_set_2d_centered.append(x-center)
	f_angle_2d=[]
	n1=pose_set_2d[20]-pose_set_2d[0]
	unit_n1 = [i / np.linalg.norm(n1) for i in n1]
	n2=[1,-unit_n1[0]/unit_n1[1]]
	unit_n2 = [i / np.linalg.norm(n2) for i in n2]
	f_angle_2d.append(get_vector_angles(pose_set_2d[20]-pose_set_2d[1],unit_n1,unit_n2))
	f_angle_2d.append(get_vector_angles(pose_set_2d[4]-pose_set_2d[1],unit_n1,unit_n2))
	f_angle_2d.append(get_vector_angles(pose_set_2d[8]-pose_set_2d[1],unit_n1,unit_n2))
	f_angle_2d.append(get_vector_angles(pose_set_2d[5]-pose_set_2d[1],unit_n1,unit_n2))
	f_angle_2d.append(get_vector_angles(pose_set_2d[9]-pose_set_2d[1],unit_n1,unit_n2))
	f_angle_2d.append(get_vector_angles(pose_set_2d[6]-pose_set_2d[1],unit_n1,unit_n2))
	f_angle_2d.append(get_vector_angles(pose_set_2d[10]-pose_set_2d[1],unit_n1,unit_n2))
	f_angle_2d.append(get_vector_angles(pose_set_2d[16]-pose_set_2d[1],unit_n1,unit_n2))
	f_angle_2d.append(get_vector_angles(pose_set_2d[12]-pose_set_2d[1],unit_n1,unit_n2))
	f_angle_2d.append(get_vector_angles(pose_set_2d[17]-pose_set_2d[1],unit_n1,unit_n2))
	f_angle_2d.append(get_vector_angles(pose_set_2d[13]-pose_set_2d[1],unit_n1,unit_n2))
	f_angle_2d.append(get_vector_angles(pose_set_2d[18]-pose_set_2d[1],unit_n1,unit_n2))
	f_angle_2d.append(get_vector_angles(pose_set_2d[14]-pose_set_2d[1],unit_n1,unit_n2))

	# fig = plt.figure(figsize=plt.figaspect(0.5))
	# ax = fig.add_subplot(1, 2, 1, projection='3d')
	# plt.title('joint map')
	# for jj in pose_set:
	# 	ax.scatter(jj[0], jj[1], jj[2], color='b')
	# ax2 = fig.add_subplot(1, 2, 2)
	# plt.title('torso')
	# for ii in pose_set_2d:
	# 	ax2.scatter(ii[0], ii[1], color='r')
	# plt.draw()
	# plt.show()
	return f_angle_2d



def gen_first_theta_phi(x, u, r, t):
	'''
	:param x: first vector
	:param u: pca u top-down
	:param r: pca r left-right
	:param t: pca t uXr
	:return:
	'''
	theta = utils.extmath.vector_angle(u, x)
	proj = utils.extmath.plane_proj(x, u)
	#get the angle between proj and t
	proj_t_angle=utils.extmath.vector_angle(proj,t)
	phi = utils.extmath.vector_angle(proj, r)
	if proj_t_angle>90:
		phi=360-phi
	return theta, phi

def gen_second_theta_phi(x,y, u, r, t):
	'''
	:param x: first vector
	:param y: second vector
	:param u: pca u top-down
	:param r: pca r left-right
	:param t: pca t uXr
	:return:
	'''
	theta = utils.extmath.vector_angle(x, y)

	proj_r = utils.extmath.plane_proj(r, x)
	proj_t = utils.extmath.plane_proj(t, x)

	#make proj_r & proj_t orthogonal
	# proj_r_norm=[i/np.linalg.norm(proj_r) for i in proj_r]
	# rt_cofficient=np.dot(proj_t, proj_r_norm) / np.dot(proj_r_norm, proj_r_norm)
	# proj_t_norm=proj_t-np.dot(rt_cofficient,proj_r_norm)

	# print np.dot(proj_r_norm,proj_t_norm)
	proj_y = utils.extmath.plane_proj(y, x)
	proj_t_angle = utils.extmath.vector_angle(proj_t, proj_y)
	phi = utils.extmath.vector_angle(proj_r, proj_y)

	if proj_t_angle>90:
		phi=360-phi
	return theta, phi

# @timeit
def export_angle_features_3d(pose_set, u, r, t):
	'''
	:param pose_set: joint maps
	:param u: pca u top-down
	:param r: pca r left-right
	:param t: pca t uXr
	:return: angles along the first and second degree joints
	'''
	ff = []

	# zero degree joints  0-11
	shoudler_left = pose_set[4] - pose_set[20]
	shoudler_right = pose_set[8] - pose_set[20]
	hip_left = pose_set[12] - pose_set[0]
	hip_right = pose_set[16] - pose_set[0]
	spine_upper = pose_set[20] - pose_set[1]
	spine_lower = pose_set[0] - pose_set[1]

	shoudler_theta_l, shoulder_phi_l = gen_first_theta_phi(shoudler_left, u, r, t)
	ff.append([shoudler_theta_l, shoulder_phi_l])

	shoudler_theta_r, shoulder_phi_r = gen_first_theta_phi(shoudler_right, u, r, t)
	ff.append([shoudler_theta_r, shoulder_phi_r])

	hip_theta_l, hip_phi_l = gen_first_theta_phi(hip_left, u, r, t)
	ff.append([hip_theta_l, hip_phi_l])

	hip_theta_r, hip_phi_r = gen_first_theta_phi(hip_right, u, r, t)
	ff.append([hip_theta_r, hip_phi_r])

	spine_theta_u, spine_phi_u = gen_first_theta_phi(spine_upper, u, r, t)
	ff.append([spine_theta_u, spine_phi_u])

	spine_theta_l, spine_phi_l = gen_first_theta_phi(spine_lower, u, r, t)
	ff.append([spine_theta_l, spine_phi_l])

	# first degree joints 12-21
	elbow_left = pose_set[5] - pose_set[4]
	elbow_right = pose_set[9] - pose_set[8]
	knee_left = pose_set[13] - pose_set[12]
	knee_right = pose_set[17] - pose_set[16]
	neck = pose_set[2] - pose_set[20]

	# elbow left & right
	elbow_theta_l, elbow_phi_l = gen_first_theta_phi(elbow_left, u, r, t)
	ff.append([elbow_theta_l, elbow_phi_l])

	elbow_theta_r, elbow_phi_r = gen_first_theta_phi(elbow_right, u, r, t)
	ff.append([elbow_theta_r, elbow_phi_r])

	# knee left & right
	knee_theta_l, knee_phi_l = gen_first_theta_phi(knee_left, u, r, t)
	ff.append([knee_theta_l, knee_phi_l])

	knee_theta_r, knee_phi_r = gen_first_theta_phi(knee_right, u, r, t)
	ff.append([knee_theta_r, knee_phi_r])

	# neck
	neck_theta, neck_phi = gen_first_theta_phi(neck, u, r, t)
	ff.append([neck_theta, neck_phi])

	# print 'ff=\n', ff

	# second degree joints 22-29
	hand_left = pose_set[6] - pose_set[5]
	hand_right = pose_set[10] - pose_set[9]
	ankle_left = pose_set[14] - pose_set[13]
	ankle_right = pose_set[18] - pose_set[17]

	# hand left & right
	hand_theta_l, hPhiL=gen_second_theta_phi(elbow_left,hand_left,u,r,t)
	ff.append([hand_theta_l, hPhiL])

	hand_theta_r, hand_phi_r = gen_second_theta_phi(elbow_right, hand_right,u,r,t)
	ff.append([hand_theta_r, hand_phi_r])

	# ankle left & right
	ankle_theta_l ,ankle_phi_l=gen_second_theta_phi(knee_left, ankle_left,u,r,t)
	ff.append([ankle_theta_l, ankle_phi_l])

	ankle_theta_r, ankle_phi_r = gen_second_theta_phi(knee_right, ankle_right,u,r,t)
	ff.append([ankle_theta_r, ankle_phi_r])

	# print 'sf=\n', sf
	return ff
