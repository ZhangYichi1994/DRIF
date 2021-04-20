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

class kde_limit():
    def __init__(self,confidence,input_data):
        self.confidence=confidence
        self.data=input_data
        self.density=kde(self.data)
    '''
    函数den_fun：计算概率分布式函数
    '''
    def den_fun(self,x):
        return self.density(x)
    '''
    函数cdf_fun：计算累积分布函数，并根据置信度创建求解方程
    '''
    def cdf_fun(self,x):
        return inter.quad(self.den_fun,-np.inf,x)[0]-self.confidence
    def fit(self):
        self.limit=fsolve(self.cdf_fun,(self.data.max()-self.data.min())/2)
        return self.limit[0]

if __name__ == '__main__':
    '''
    CSTH数据集
    train_data = loadmat('origin-data')['data']
    test_data=train_data
    '''
    # train_data = loadmat('origin-data')['data']
    # test_data=train_data
    '''
    UCI数据集
    fname = "Iris.txt"
    origin_data=np.loadtxt(fname, delimiter=",")
    train_data=origin_data[:,1:]
    test_data=origin_data[:,1:]
    '''

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
    print(train_label)
    print(test_label)

    '''
    训练数据与测试数据标准化
    '''
    train_mean=np.mean(train_data,axis=0)           # 归一化均值
    train_std=np.std(train_data,axis=0)             # 归一化方差
    trainData=(train_data-train_mean)/train_std    # 训练集归一化

    test_mean = train_mean
    test_std = train_std

    #参数设置
    net_sum=[]  #集合所有已经训练好的SOM网络
    element_in_cluster=200  #每个类别中元素的个数
    feature_number=train_data.shape[1]  #训练数据的特征个数
    # train_data_split=train_split(train_data,element_in_cluster)
     
    max_iter=500

    


    #网络训练阶段
    learnt_net={}
    size = math.ceil(np.sqrt(5 * np.sqrt(train_data.shape[0])))
    som = MiniSom(size, size, feature_number, sigma=2, learning_rate=0.5,
                    neighborhood_function='bubble')
    som.train_batch(trainData,max_iter,verbose=True)
    learnt_net['net']=som
    tmp=[]
    for k in range(len(trainData)):
        tmp.append(som.quantization_error(trainData[k:k+1,:]))
    print(tmp)
    learnt_net['mean_quantization_error']=np.mean(tmp)  #计算网络训练时的平均量化误差 （等同于论文中的Eq-training）
    learnt_net['max_quantization_error']=np.max(tmp)    #计算网络训练时的最大量化误差 (等同于论文中的d_max-training)

    cal_limit=kde_limit(0.99,np.array(tmp))   #创建一个kde控制限计算的对象（创建时要输入两个参数：1.置信度 2.输入数据）
    limit2=cal_limit.fit()              #调用该对象中计算功能（无需输入其他参数，自动进行计算）
    learnt_net['kde_limit'] = limit2
    print(limit2)

    learnt_net['test_mean'] = test_mean
    learnt_net['test_std'] = test_std

    # 需要保存的数据：SOM网络learnt_net、样本的均值test_mean、样本的方差test_std
    with open('som.p', 'wb') as outfile:
        pickle.dump(learnt_net, outfile)


