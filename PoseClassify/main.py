import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import estimators
import tensorflow.contrib.layers as tfl

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
    posX.append(posSet.ravel())
posX=np.array(posX)
posY=np.array([[1,0] for x in range(len(posX))])

#load wrong pose
data = loadData('./kinect_data/chestapWrong.txt')

negNum=len(data)/25
negX=[]
negSet=np.zeros((25,3))

for j in xrange(0,negNum):
    for i in xrange(0,25):
        #data[0][1][2:-1]
        negSet[i]=[float(data[i][1][2:-1]),float(data[i][2][2:-1]),float(data[i][3][2:-1])]
    negX.append(negSet.ravel())
negX = np.array(negX)
negY=np.array([[0,1] for x in range(len(negX))])

trainx=np.concatenate((posX,negX),axis=0)
trainy=np.concatenate((posY,negY),axis=0)

sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,75])
W=tf.Variable(tf.zeros([75,2]),name="joints")
b=tf.Variable(tf.zeros([2]),name="jonitBias")
y=tf.nn.softmax(tf.matmul(x,W)+b)
y_=tf.placeholder(tf.float32,[None,2])
crossEntropyLoss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(crossEntropyLoss)
tf.global_variables_initializer().run()

sess.run(train,{x:trainx,y_:trainy})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, crossEntropyLoss], {x: trainx, y_: trainy})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
testx=np.reshape(trainx[15],(1,75))
yPredict=np.matmul(testx,sess.run(W))+sess.run(b)
print yPredict






























