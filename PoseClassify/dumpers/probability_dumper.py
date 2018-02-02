import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

#feature_pos_all=np.load('.\\feature_pos.txt')
feature_pos_hug=np.load('.\\feature_pos_hug.txt')
feature_pos_cross=np.load('.\\feature_pos_akimbo.txt')

feature_num=51


fig, (ax1) = plt.subplots(1, 1, sharex=True, sharey=True)

sum_feature_hug=np.zeros(feature_num)
avg_feature_hug=np.zeros(feature_num)
for f in range(0,len(feature_pos_hug)):
	for p in range(0,len(feature_pos_hug[f])):
		sum_feature_hug[p]+=feature_pos_hug[f][p]
for m in range(0,feature_num):
	avg_feature_hug[m]=sum_feature_hug[m]/len(feature_pos_hug)
ax1.plot(range(0, len(avg_feature_hug)), avg_feature_hug, 'go--',alpha=0.3)
for f in range(0,len(feature_pos_hug)):
		ax1.plot(range(0, len(feature_pos_hug[f])), feature_pos_hug[f], 'go',alpha=0.005)

fig, (ax2) = plt.subplots(1, 1, sharex=True, sharey=True)
sum_feature_cross=np.zeros(feature_num)
avg_feature_akimbo=np.zeros(feature_num)
for f in range(0,len(feature_pos_cross)):
	for p in range(0,len(feature_pos_cross[f])):
		sum_feature_cross[p]+=feature_pos_cross[f][p]
for m in range(0,feature_num):
	avg_feature_akimbo[m]=sum_feature_cross[m]/len(feature_pos_cross)
ax2.plot(range(0, len(avg_feature_akimbo)), avg_feature_akimbo, 'ro--',alpha=0.3)
for f in range(0,len(feature_pos_cross)):
		ax2.plot(range(0, len(feature_pos_cross[f])), feature_pos_cross[f], 'ro',alpha=0.005)

#sum_feature_all=np.zeros(feature_num)
#avg_feature_all=np.zeros(feature_num)
#for f in range(0,len(feature_pos_all)):
#	for p in range(0,len(feature_pos_all[f])):
#		sum_feature_all[p]+=feature_pos_all[f][p]
#for m in range(0,feature_num):
#	avg_feature_all[m]=sum_feature_all[m]/len(feature_pos_all)
#ax1.plot(range(0, len(avg_feature_all)), avg_feature_all, 'bo--',alpha=0.7)
#for f in range(0,len(feature_pos_all)):
#		ax1.plot(range(0, len(feature_pos_all[f])), feature_pos_all[f], 'ro',alpha=0.01)

outfile_akimbo = open("avg_feature_akimbo.txt", 'ab')
np.save(outfile_akimbo, avg_feature_akimbo)

outfile_hug = open("avg_feature_hug.txt", 'ab')
np.save(outfile_hug, avg_feature_hug)

for ax in ax1,ax2:
	ax.set_xticks(range(0,feature_num))
	ax.xaxis.grid(True)
fig.suptitle('probability and average distribution')
fig.autofmt_xdate()
plt.show()


