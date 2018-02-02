import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def get_feature_avg():
    avg_feature_akimbo=np.load('.\\avg_feature_akimbo.txt')
    avg_feature_hug=np.load('.\\avg_feature_hug.txt')
    return avg_feature_akimbo,avg_feature_hug