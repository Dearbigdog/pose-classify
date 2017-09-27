import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn.decomposition import PCA
import utils.modules as modules
import os
import tempfile

'''
generate the features and store them instead of calculate every time
'''
def gen_feature_file(features_file, source_file):
    features = []
    # determine whether we need to generate the files
    if not os.path.isfile(features_file):
        outfile = open(features_file, 'ab')
        # load pos data and convert them to theta and phi
        for p in source_file:
            print 'loading... path=', p
            pos_data = modules.load_data(p)
            data_x, torsos = modules.get_torso(pos_data)

            for i in range(0, torsos.shape[0]):
                print 'total torso= {0},executing torso:{1}'.format(torsos.shape[0], i)
                u, r, t = modules.get_torso_pca(torsos[i])
                f = modules.export_features(data_x[i], u, r, t)
                features.append(f)

        features = np.array(features)
        print features.shape
        np.save(outfile, features)
        outfile.close()
        print 'finished...'
    else:
        print 'loading features...',features_file
        outfile = open(features_file, 'rb')
        features = np.load(outfile)
        outfile.close()
        print features.shape
    return features

feature_pos_file='f_pos.txt'
data_pos_files=['./kinect_data/jointPos_Richard_Pos.txt','./kinect_data/jointPos_Dexter_Pos.txt','./kinect_data/jointPos_Jay_Pos.txt']
feature_neg_file='f_neg.txt'
data_neg_files=['./kinect_data/jointPos_Richard_Neg.txt','./kinect_data/jointPos_Dexter_Neg.txt','./kinect_data/jointPos_Jay_Neg.txt']

data_pos=gen_feature_file(feature_pos_file,data_pos_files)
data_neg=gen_feature_file(feature_neg_file,data_neg_files)

