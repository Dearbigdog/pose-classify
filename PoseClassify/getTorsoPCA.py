import  numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
print torsoMatrix,torsoMatrix.shape

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, Z)
plt.show()
















