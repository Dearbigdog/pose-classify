import  numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn.decomposition import PCA



def loadData(fname):
    ''' load the file using std open'''
    f = open(fname,'r')
    data = []
    for line in f.readlines():
        data.append(line.replace('\n','').split(' '))
    f.close()
    return data

#load correct pose
data = loadData('./kinect_data/chestap.txt')

posNum=len(data)/25
posX=[]
posSet=np.zeros((25,3))
for j in xrange(0,posNum):
    for i in xrange(0,25):
        #data[0][1][2:-1]
        posSet[i]=np.array([float(data[i][1][2:-1]),float(data[i][2][2:-1]),float(data[i][3][2:-1])])
    posX.append(posSet)
posX=np.array(posX)
posY=np.array([[1,0] for x in range(len(posX))])
#jonit 0
print posX[0][0]

#8,20,4
#1
#16 0 12
X=[]
Y=[]
Z=[]
torsoIndices=[8,20,4,1,16,0,12]
#torsoIndices=[x for x in range(0,25)]
print torsoIndices
torsoMatrix=[]
for i in torsoIndices:
    torsoMatrix.append(posX[0][i])
    X.append(posX[0][i][0])
    Y.append(posX[0][i][1])
    Z.append(posX[0][i][2])

torsoMatrix=np.array(torsoMatrix)
print 'torsoMatrix shape=',torsoMatrix.shape


#sklearn
def doPCA():
    pca=PCA(n_components=3)
    pca.fit(torsoMatrix)
    return pca
pca=doPCA()
print pca.explained_variance_ratio_
firstPc=pca.components_[0]
secondPc=pca.components_[1]

print 'first pc=',firstPc
print 'second pc=',secondPc

transformedData=pca.transform(torsoMatrix)
print 'transformed data shape=',transformedData.shape
fig2 = plt.figure()
ax2 = Axes3D(fig2)

for ii,jj in zip(transformedData,torsoMatrix):
    ax2.scatter(firstPc[0]*ii[0],firstPc[1]*ii[0],firstPc[2]*ii[0],color='r')
    ax2.scatter(secondPc[0] * ii[1], secondPc[1] * ii[1], secondPc[2] * ii[1], color='c')
    ax2.scatter(jj[0],jj[1],jj[2],color='b')
plt.draw()


#tensorflow
factors=2
sess=tf.Session()
# SVD
torsoPca=torsoMatrix-np.mean(torsoMatrix,axis=0)
#torsoPca=np.matmul(torsoPca,np.transpose(torsoPca))
St, Ut, Vt = tf.svd(torsoPca,full_matrices=False)
print 'Ut=\n',sess.run(Ut)
print 'St=\n',sess.run(St)
print 'Vt=\n',sess.run(Vt)

# Compute reduced matrices
Sk = tf.diag(St)[0:factors, 0:factors]
Vk = Vt[:, 0:factors]
Uk=Ut[0:factors,:]

print 'Vk=\n',sess.run(Vk)

# Compute Su and Si
#Su = tf.matmul(Uk, tf.sqrt(Sk))
#Si = tf.matmul(tf.sqrt(Sk), Vk)

# Compute user average rating
torsoNew = sess.run(tf.matmul(torsoPca,Vk))
print 'torso new.shape',torsoNew.shape

u=sess.run(Vk)[:,0]
r=sess.run(Vk)[:,1]
t=np.cross(r,u)

print 'u=\n',u
print 'v=\n',r

fig = plt.figure()
ax = Axes3D(fig)

for ii,jj in zip(torsoNew,torsoMatrix):
    ax.scatter(u[0]*ii[0],u[1]*ii[0],u[2]*ii[0],color='r')
    ax.scatter(r[0] * ii[1], r[1] * ii[1], r[2] * ii[1], color='c')
    ax.scatter(jj[0],jj[1],jj[2],color='b')
plt.draw()

plt.show()
