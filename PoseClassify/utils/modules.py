import numpy as np
import tensorflow as tf

def loadData(fname):
    ''' load the file using std open'''
    f = open(fname, 'r')
    data = []
    for line in f.readlines():
        data.append(line.replace('\n', '').split(' '))
    f.close()
    return data


def getTorso(data):
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
    return torsoMatrix

def getTorsoPca(torsoMat):
    # tensorflow
    factors = 2
    sess = tf.Session()
    # SVD
    # Center the points
    torsoPca = torsoMat - np.mean(torsoMat, axis=0)
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

    # t = u X r
    t = np.cross(u, r)

    print 'u=', u
    print 'r=', r
    print 't=', t
    return u,r,t

def exportFeatures(pos):
    ff = []
    sf = []
    return ff, sf
