import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

test_x = np.load('../test_x.txt')
test_y = np.load('../test_y.txt')
logits=np.load('../test_logits.txt')
test_x_path=np.load('../test_x_path.txt')



test_y_rename=[]
for y in test_y:
	if y[0]==1:
		test_y_rename.append(True)
	else:
		test_y_rename.append(False)
test_x_data = np.load('../test_x_data.txt')
test_correct_pred = np.load('../test_correct_pred.txt')
index_tracer=[]
for i in xrange(len(test_correct_pred)):
	if test_correct_pred[i]==False:
		index_tracer.append(i)


#softmax demo
logits_modified=[]
logits_copy=np.copy(logits)
for l in logits_copy:
	l =l-np.max(l)
	p=np.exp(l)/np.sum(np.exp(l))
	logits_modified.append(p)

test_correct_pred_rename=[]
for p in test_correct_pred:
	if p==True:
		test_correct_pred_rename.append('correct')
	else:
		test_correct_pred_rename.append('incorrect')

# get torso 2d
torsoIndices = [8, 20, 4, 1, 16, 0, 12]

joint_map=test_x_data[8]

torso_map = []
for i in torsoIndices:
	torso_map.append(joint_map[i])

joint_map_mean = joint_map - np.mean(joint_map, axis=0)
# u,r,t=modules.get_torso_pca(joint_map)


def doPCA(joint_map_mean):
	pca = PCA(n_components=2)
	pca.fit(joint_map_mean)
	return pca
pca = doPCA(joint_map_mean)

firstPc = pca.components_[0]
secondPc = pca.components_[1]


transformedData = pca.transform(joint_map_mean)

for i in index_tracer:
	fig = plt.figure(figsize=plt.figaspect(1))

	ax = fig.add_subplot(2, 2, 1, projection='3d')
	plt.title(
		'Prediction : {0}\n loss={1}\n softmax loss={2}\n truth={3}'.format(test_correct_pred_rename[i], logits[i],
		                                                                    logits_modified[i], test_y[i]))
	for j in xrange(len(test_x_data[i])):
		if j==1 or j==4 or j==8 or j==16 or j==12or j==20 or j==0:
			ax.scatter(test_x_data[i][j][0], test_x_data[i][j][1], test_x_data[i][j][2], color='r')
		elif j==10 or j==6 or j==7 or j==11 or j==23 or j==21 or j==9 or j==5 or j==22 or j==24:
			ax.scatter(test_x_data[i][j][0], test_x_data[i][j][1], test_x_data[i][j][2], color='m')
		elif j==2 or j==3 :
			ax.scatter(test_x_data[i][j][0], test_x_data[i][j][1], test_x_data[i][j][2], color='g')
		else:
			ax.scatter(test_x_data[i][j][0], test_x_data[i][j][1], test_x_data[i][j][2], color='b')

	ax2 = fig.add_subplot(2, 2, 2)
	plt.title('torso projection')
	for ii in transformedData:
		ax2.scatter(ii[0], ii[1], color='r')


	fig.add_subplot(2, 2, 3)
	plt.title('43 features distribution')
	plt.plot(xrange(0, len(test_x[i])), test_x[i], 'ro--')

	fig.add_subplot(2,2, 4)
	plt.axis('off')
	plt.title('corresponding rgb : {0}'.format(test_x_path[i]))
	rgb_p1=test_x_path[i].split('/')
	rgb_p2=rgb_p1[5][0:8]+rgb_p1[5][9:17]+'_color.bmp'
	rgb_path=rgb_p1[0]+'/'+rgb_p1[1]+'/'+rgb_p1[2]+'/'+rgb_p1[3]+'/'+rgb_p1[4]+'/'+rgb_p2
	img = mpimg.imread(rgb_path)
	plt.imshow(img)


	plt.draw()
	mng = plt.get_current_fig_manager()
	mng.window.showMaximized()
	plt.show()


for i in xrange(0, len(test_correct_pred), 1):
	fig = plt.figure(figsize=plt.figaspect(1))

	ax = fig.add_subplot(2, 2, 1, projection='3d')
	plt.title(
		'Prediction : {0}\n loss={1}\n softmax loss={2}\n truth={3}'.format(test_correct_pred_rename[i], logits[i],
		                                                                    logits_modified[i], test_y[i]))
	for j in xrange(len(test_x_data[i])):
		if j==1 or j==4 or j==8 or j==16 or j==12or j==20 or j==0:
			ax.scatter(test_x_data[i][j][0], test_x_data[i][j][1], test_x_data[i][j][2], color='r')
		elif j==10 or j==6 or j==7 or j==11 or j==23 or j==21 or j==9 or j==5 or j==22 or j==24:
			ax.scatter(test_x_data[i][j][0], test_x_data[i][j][1], test_x_data[i][j][2], color='m')
		elif j==2 or j==3 :
			ax.scatter(test_x_data[i][j][0], test_x_data[i][j][1], test_x_data[i][j][2], color='g')
		else:
			ax.scatter(test_x_data[i][j][0], test_x_data[i][j][1], test_x_data[i][j][2], color='b')

	ax2 = fig.add_subplot(2, 2, 2)
	plt.title('torso projection')
	for ii in transformedData:
		ax2.scatter(ii[0], ii[1], color='r')


	fig.add_subplot(2, 2, 3)
	plt.title('43 features distribution')
	plt.plot(xrange(0, len(test_x[i])), test_x[i], 'ro--')

	fig.add_subplot(2,2, 4)
	plt.axis('off')
	plt.title('corresponding rgb : {0}'.format(test_x_path[i]))
	rgb_p1=test_x_path[i].split('/')
	rgb_p2=rgb_p1[5][0:8]+rgb_p1[5][9:17]+'_color.bmp'
	rgb_path=rgb_p1[0]+'/'+rgb_p1[1]+'/'+rgb_p1[2]+'/'+rgb_p1[3]+'/'+rgb_p1[4]+'/'+rgb_p2
	img = mpimg.imread(rgb_path)
	plt.imshow(img)


	plt.draw()
	mng = plt.get_current_fig_manager()
	mng.window.showMaximized()
	plt.show()
