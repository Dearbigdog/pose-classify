import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils.modules as modules
from sklearn.decomposition import PCA

test_x = np.load('../test_x.txt')
test_y = np.load('../test_y.txt')
test_y_rename=[]
for y in test_y:
	if y[0]==1:
		test_y_rename.append(True)
	else:
		test_y_rename.append(False)
test_x_data = np.load('../test_x_data.txt')
test_correct_pred = np.load('../test_correct_pred.txt')

# get torso 7 points
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
print(pca.explained_variance_ratio_)
firstPc = pca.components_[0]
secondPc = pca.components_[1]

print('first pc=', firstPc)
print('second pc=', secondPc)

transformedData = pca.transform(joint_map_mean)
print('transformed data shape=', transformedData.shape)




fig = plt.figure(figsize=plt.figaspect(0.3))

ax = fig.add_subplot(1,3,1, projection='3d')
plt.title('joint map')
for jj in joint_map:
	ax.scatter(jj[0], jj[1], jj[2], color='b')
# ax2=fig.add_subplot(1, 3, 2, projection='3d')
# plt.title('torso')
# for jj in torso_map:
# 	ax2.scatter(jj[0], jj[1], jj[2], color='r')

ax2=fig.add_subplot(1, 3, 2)
plt.title('torso')
for ii in transformedData:
	ax2.scatter(ii[0], ii[1], color='r')
	# ax2.scatter(firstPc[0]*ii[0],firstPc[1]*ii[0],firstPc[2]*ii[0],color='g')
	# ax2.scatter(secondPc[0] * ii[1], secondPc[1] * ii[1], secondPc[2] * ii[1], color='c')

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
for ii,jj in zip(transformedData,joint_map_mean):
	ax3.scatter(firstPc[0]*ii[0],firstPc[1]*ii[0],firstPc[2]*ii[0],color='r')
	ax3.scatter(secondPc[0] * ii[1], secondPc[1] * ii[1], secondPc[2] * ii[1], color='c')
	ax3.scatter(jj[0],jj[1],jj[2],color='b')
plt.draw()

plt.show()