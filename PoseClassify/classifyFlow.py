import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn.decomposition import PCA
import utils.extmath


def loadData(fname):
    ''' load the file using std open'''
    f = open(fname, 'r')
    data = []
    for line in f.readlines():
        data.append(line.replace('\n', '').split(' '))
    f.close()
    return data


# load correct pose
data = loadData('./kinect_data/jointPos_Richard_Pos.txt')

posNum = len(data) / 25
posX = []
posSet = np.zeros((25, 3))
for j in xrange(0, posNum):
    for i in xrange(0, 25):
        # data[0][1][2:-1]
        posSet[i] = np.array([float(data[i][1][2:-1]), float(data[i][2][2:-1]), float(data[i][3][2:-1])])
    posX.append(posSet)
posX = np.array(posX)
posY = np.array([[1, 0] for x in range(len(posX))])


# get torso 7 points
torsoIndices = [8, 20, 4, 1, 16, 0, 12]
torsoMatrix = []
for i in torsoIndices:
    torsoMatrix.append(posX[0][i])

torsoMatrix = np.array(torsoMatrix)
print 'torsoMatrix shape=', torsoMatrix.shape


# sklearn
def doPCA():
    pca = PCA(n_components=2)
    pca.fit(torsoMatrix)
    return pca


pca = doPCA()
print pca.explained_variance_ratio_
firstPc = pca.components_[0]
secondPc = pca.components_[1]

print 'first pc=', firstPc
print 'second pc=', secondPc

transformedData = pca.transform(torsoMatrix)
print 'transformed data shape=', transformedData.shape
# fig2 = plt.figure()
# ax2 = Axes3D(fig2)
# for ii,jj in zip(transformedData,torsoMatrix):
#     ax2.scatter(firstPc[0]*ii[0],firstPc[1]*ii[0],firstPc[2]*ii[0],color='r')
#     ax2.scatter(secondPc[0] * ii[1], secondPc[1] * ii[1], secondPc[2] * ii[1], color='c')
#     ax2.scatter(jj[0],jj[1],jj[2],color='b')
# plt.draw()


# tensorflow
factors = 2
sess = tf.Session()
# SVD
# Center the points
torsoPca = torsoMatrix - np.mean(torsoMatrix, axis=0)
# torsoPca=np.matmul(torsoPca,np.transpose(torsoPca))
St, Ut, Vt = tf.svd(torsoPca, full_matrices=False)
print 'Ut=\n', sess.run(Ut)
print 'St=\n', sess.run(St)
print 'Vt=\n', sess.run(Vt)

# Compute reduced matrices
Sk = tf.diag(St)[0:factors, 0:factors]
Vk = Vt[:, 0:factors]
Uk = Ut[0:factors, :]

print 'Vk=\n', sess.run(Vk)

# Compute Su and Si
# Su = tf.matmul(Uk, tf.sqrt(Sk))
# Si = tf.matmul(tf.sqrt(Sk), Vk)

# Compute user average rating
torsoNew = sess.run(tf.matmul(torsoPca, Vk))
print 'torso new.shape', torsoNew.shape

u = sess.run(Vk)[:, 0]
r = sess.run(Vk)[:, 1]

# u top-down
max_abs_cols = np.argmax(np.abs(u))
signs = np.sign(u[max_abs_cols])
u *= -signs
# r left right

# t = u*r
t = np.cross(u, r)

print 'u=', u
print 'r=', r
print 't=', t

fig = plt.figure()
ax = Axes3D(fig)
for ii, jj, kk in zip(torsoNew, torsoMatrix, torsoPca):
    ax.scatter(u[0] * ii[0], u[1] * ii[0], u[2] * ii[0], color='r')
    ax.scatter(r[0] * ii[1], r[1] * ii[1], r[2] * ii[1], color='c')
    ax.scatter(jj[0], jj[1], jj[2], color='b')
    ax.scatter(kk[0], kk[1], kk[2], color='m')
plt.draw()

# first degree joints
ff = []

vElbowLeft = posX[0][5] - posX[0][4]
vElbowRight = posX[0][9] - posX[0][8]
vKneeLeft = posX[0][13] - posX[0][12]
vKneeRight = posX[0][17] - posX[0][16]
vNeck = posX[0][2] - posX[0][20]

# elbow left & right
eThetaL = utils.extmath.vAngle(u, vElbowLeft)
print 'left elbow theta=', eThetaL
eProjL = utils.extmath.planeProject(vElbowLeft, u)
ePhiL = utils.extmath.vAngle(eProjL, r)
print 'left elbow phi=', ePhiL
ff.append([eThetaL, ePhiL])

eThetaR = utils.extmath.vAngle(u, vElbowRight)
print 'right elbow theta=', eThetaR
eProjR = utils.extmath.planeProject(vElbowRight, u)
ePhiR = utils.extmath.vAngle(eProjR, r)
print 'right elbow phi=', ePhiR
ff.append([eThetaR, ePhiR])

# knee left & right
kThetaL = utils.extmath.vAngle(u, vKneeLeft)
print 'left knee theta=', kThetaL
kProjL = utils.extmath.planeProject(vKneeLeft, u)
kPhiL = utils.extmath.vAngle(kProjL, r)
print 'left knee phi=', kPhiL
ff.append([kThetaL, kPhiL])

kThetaR = utils.extmath.vAngle(u, vKneeRight)
print 'right knee theta=', kThetaR
kProjR = utils.extmath.planeProject(vKneeRight, u)
kPhiR = utils.extmath.vAngle(kProjR, r)
print 'right knee phi=', kPhiR
ff.append([kThetaR, kPhiR])

# neck
nTheta = utils.extmath.vAngle(u, vNeck)
print 'neck theta=', nTheta
nProj = utils.extmath.planeProject(vNeck, u)
nPhi = utils.extmath.vAngle(nProj, r)
print 'neck phi=', nPhi
ff.append([nTheta, nPhi])

# second degree joints

vHandLeft = posX[0][6] - posX[0][5]
vHandRight = posX[0][10] - posX[0][9]
vAnkleLeft = posX[0][14] - posX[0][13]
vAnkelRight = posX[0][18] - posX[0][17]

# hand left & right
hThetaL = utils.extmath.vAngle(vElbowLeft, vHandLeft)
print 'left hand theta=', hThetaL
rProj = utils.extmath.planeProject(r, vElbowLeft)
vhandLProj = utils.extmath.planeProject(vHandLeft, vElbowLeft)
hPhiL = utils.extmath.vAngle(rProj, vhandLProj)
print 'left hand phi=', hPhiL
ff.append([hThetaL, hPhiL])

hThetaR = utils.extmath.vAngle(vElbowRight, vHandRight)
print 'right hand theta=', hThetaR
rProj = utils.extmath.planeProject(r, vElbowRight)
vhandRProj = utils.extmath.planeProject(vHandRight, vElbowRight)
hPhiR = utils.extmath.vAngle(rProj, vhandRProj)
print 'right hand phi=', hPhiR
ff.append([hThetaR, hPhiR])

# ankle left & right
aThetaL = utils.extmath.vAngle(vKneeLeft, vAnkleLeft)
print 'left ankle theta=', aThetaL
rProj = utils.extmath.planeProject(r, vKneeLeft)
vAnkleLProj = utils.extmath.planeProject(vAnkleLeft, vKneeLeft)
aPhiL = utils.extmath.vAngle(rProj, vAnkleLProj)
print 'left ankle phi=', aPhiL
ff.append([aThetaL, aPhiL])

aThetaR = utils.extmath.vAngle(vKneeRight, vAnkelRight)
print 'right ankle theta=', aThetaR
rProj = utils.extmath.planeProject(r, vKneeRight)
vAnkleRProj = utils.extmath.planeProject(vAnkelRight, vKneeRight)
aPhiR = utils.extmath.vAngle(rProj, vAnkleRProj)
print 'right ankle phi=', aPhiR
ff.append([aThetaR, aPhiR])

print 'ff=\n', ff
ff=np.array(ff)
print ff.shape
# fig = plt.figure()
# ax = Axes3D(fig)
# for jj in dfMat:
#     ax.scatter(jj[0],jj[1],jj[2],color='b')
# plt.draw()







plt.show()
