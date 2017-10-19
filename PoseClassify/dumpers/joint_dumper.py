import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

for i in xrange(0, len(test_correct_pred), 3):
	fig = plt.figure(figsize=plt.figaspect(0.4))

	ax = fig.add_subplot(2, 3, 1, projection='3d')
	plt.title('predict={0}, truth={1}'.format(test_correct_pred[i],test_y_rename[i]))
	for jj in test_x_data[i]:
		ax.scatter(jj[0], jj[1], jj[2], color='b')
	fig.add_subplot(2, 3, 4)
	plt.plot(xrange(0, len(test_x[i])), test_x[i], 'ro--')

	ax = fig.add_subplot(2, 3, 2, projection='3d')
	plt.title('predict={0}, truth={1}'.format(test_correct_pred[i+1],test_y_rename[i+1]))
	for jj in test_x_data[i+1]:
		ax.scatter(jj[0], jj[1], jj[2], color='b')
	fig.add_subplot(2, 3, 5)
	plt.plot(xrange(0, len(test_x[i+1])), test_x[i+1], 'ro--')

	ax = fig.add_subplot(2, 3, 3, projection='3d')
	plt.title('predict={0}, truth={1}'.format(test_correct_pred[i+2],test_y_rename[i+2]))
	for jj in test_x_data[i+2]:
		ax.scatter(jj[0], jj[1], jj[2], color='b')
	fig.add_subplot(2, 3, 6)
	plt.plot(xrange(0, len(test_x[i+2])), test_x[i+2], 'ro--')



	plt.draw()
	plt.show()
