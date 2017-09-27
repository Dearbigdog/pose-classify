import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn.decomposition import PCA
import utils.modules as modules
import os


features_file='features_richard.dat'

features = []

#determine whether we need to generate the files
if not os.path.isfile(features_file):
    print 'extract features...'
    #load pos data and convert them to theta and phi
    pos_data = modules.load_data('./kinect_data/jointPos_Richard_Pos.txt')
    data_x,torsos=modules.get_torso(pos_data)
    print torsos.shape[0]
    # print data_x.shape

    for i in range(0,torsos.shape[0]):
        print 'executing...',i
        u,r,t=modules.get_torso_pca(data_x[i])
        f=modules.export_features(data_x[i],u,r,t)
        features.append(f)

    features=np.array(features)
    print features.shape
    features.tofile('features_richard.dat','wb')
    print 'finished...'
else:
    print 'loading features...'
    fh = open("features_richard.dat", "rb")
    features=np.fromfile(fh)

print features.shape