from minisom import MiniSom
import math
import numpy as np
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import pickle
from sklearn import datasets
from sklearn.metrics import classification_report
from scipy.stats import gaussian_kde as kde
from scipy.optimize import fsolve
from scipy import integrate as inter
import numpy as np
import random
import pickle

def classify(som, data, test_mean, test_std, model):
    data = (data - test_mean) / test_std
    label = []
    if model == 0:
        for i in range (len(data)):
            e_q=som['net'].quantization_error(data[i:i+1,:])
            if e_q <= som['mean_quantization_error']:
                test_class=0
                label.append(test_class)
            else:
                test_class = 1
                label.append(test_class)
    if model == 1:
        for i in range(len(data)):
            e_q = som['net'].quantization_error(data[i:i+1, :])
            if e_q <= som['kde_limit']:
                test_class = 0
                label.append(test_class)
            else:
                test_class = 1
                label.append(test_class)
    return label

if __name__ == '__main__':
    '''
    Iris 数据集
    '''
    iris = datasets.load_iris()
    # iris = random.shuffle(iris)
    # iris = iris.shuffle()
    train_data = iris.data[0:100,:]
    train_label = iris.target[0:100]
    test_data = iris.data[100:,:]
    test_label = iris.target[100:]
    test_data = np.row_stack((test_data, np.array([0,0,0,0])))

    with open('som.p', 'rb') as infile:
        learnt_net = pickle.load(infile)

    # 网络测试
    model = 1
    y_predict = classify(learnt_net, test_data, test_mean, test_std, model)
    print(y_predict)
    y_train_predict = classify(learnt_net, train_data, test_mean, test_std, model)
    print(y_train_predict)