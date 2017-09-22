import numpy as np
import tensorflow as tf
import utils.extmath

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

# get u v r from torso mat
def getTorsoPca(torsoMat,factors=2):
    # tensorflow
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

def exportFeatures(posSet,u,r,t):
    # first degree joints
    ff = []

    vElbowLeft = posSet[5] - posSet[4]
    vElbowRight = posSet[9] - posSet[8]
    vKneeLeft = posSet[13] - posSet[12]
    vKneeRight = posSet[17] - posSet[16]
    vNeck = posSet[2] - posSet[20]

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

    print 'ff=\n', ff

    # second degree joints
    sf = []
    vHandLeft = posSet[6] - posSet[5]
    vHandRight = posSet[10] - posSet[9]
    vAnkleLeft = posSet[14] - posSet[13]
    vAnkelRight = posSet[18] - posSet[17]

    # hand left & right
    hThetaL = utils.extmath.vAngle(vElbowLeft, vHandLeft)
    print 'left hand theta=', hThetaL
    rProj = utils.extmath.planeProject(r, vElbowLeft)
    vhandLProj = utils.extmath.planeProject(vHandLeft, vElbowLeft)
    hPhiL = utils.extmath.vAngle(rProj, vhandLProj)
    print 'left hand phi=', hPhiL
    sf.append([hThetaL, hPhiL])

    hThetaR = utils.extmath.vAngle(vElbowRight, vHandRight)
    print 'right hand theta=', hThetaR
    rProj = utils.extmath.planeProject(r, vElbowRight)
    vhandRProj = utils.extmath.planeProject(vHandRight, vElbowRight)
    hPhiR = utils.extmath.vAngle(rProj, vhandRProj)
    print 'right hand phi=', hPhiR
    sf.append([hThetaR, hPhiR])

    # ankle left & right
    aThetaL = utils.extmath.vAngle(vKneeLeft, vAnkleLeft)
    print 'left ankle theta=', aThetaL
    rProj = utils.extmath.planeProject(r, vKneeLeft)
    vAnkleLProj = utils.extmath.planeProject(vAnkleLeft, vKneeLeft)
    aPhiL = utils.extmath.vAngle(rProj, vAnkleLProj)
    print 'left ankle phi=', aPhiL
    sf.append([aThetaL, aPhiL])

    aThetaR = utils.extmath.vAngle(vKneeRight, vAnkelRight)
    print 'right ankle theta=', aThetaR
    rProj = utils.extmath.planeProject(r, vKneeRight)
    vAnkleRProj = utils.extmath.planeProject(vAnkelRight, vKneeRight)
    aPhiR = utils.extmath.vAngle(rProj, vAnkleRProj)
    print 'right ankle phi=', aPhiR
    sf.append([aThetaR, aPhiR])

    print 'sf=\n', sf
    return ff, sf
