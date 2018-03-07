import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import neural_network.restore_model as rm


def get_feature_avg():
    avg_feature_akimbo=np.load('.\\avg_feature_akimbo.txt')
    avg_feature_hug=np.load('.\\avg_feature_hug.txt')
    return avg_feature_akimbo,avg_feature_hug

def test_two_way(array_x,hug_para,akimbo_para):
	f_combine_hug,out_layer_hug=rm.neural_network_classify(array_x,hug_para)
	f_combine_akimbo,out_layer_akimbo=rm.neural_network_classify(array_x,akimbo_para)
	if (np.argmax(out_layer_hug)==0):
		return f_combine_hug,out_layer_hug
	if (np.argmax(out_layer_akimbo)==0):
		return f_combine_akimbo,out_layer_akimbo
	else:
		return f_combine_hug,out_layer_hug