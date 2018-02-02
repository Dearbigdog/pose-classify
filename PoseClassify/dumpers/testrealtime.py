import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

feature_pos_all=np.load('.\\feature_pos_all.txt')
feature_pos_hug=np.load('.\\feature_pos_hug.txt')
feature_pos_cross=np.load('.\\feature_pos_cross.txt')

plt.axis()
plt.ion()

sum_feature_hug=np.zeros(43)
avg_feature_hug=np.zeros(43)
for f in range(0,len(feature_pos_hug)):
	for p in range(0,len(feature_pos_hug[f])):
		sum_feature_hug[p]+=feature_pos_hug[f][p]
for m in range(0,43):
	avg_feature_hug[m]=sum_feature_hug[m]/len(feature_pos_hug)



for f in range(0,len(feature_pos_hug)):
    plt.plot(range(0, len(avg_feature_hug)), avg_feature_hug, 'ro--',alpha=0.3)
    plt.plot(range(0, len(feature_pos_hug[f])), feature_pos_hug[f], 'go--',alpha=1)
    plt.pause(0.05)
    plt.clf()
while True:
    plt.pause(0.05)