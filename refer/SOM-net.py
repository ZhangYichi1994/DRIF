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

    # 需要保存的数据：SOM网络learnt_net、样本的均值test_mean、样本的方差test_std


    # 网络测试
    model = 1
    y_predict = classify(learnt_net, test_data, test_mean, test_std, model)
    print(y_predict)
    y_train_predict = classify(learnt_net, train_data, test_mean, test_std, model)
    print(y_train_predict)
    print(classification_report(test_label, y_predict[0:-1]))







    #网络测试阶段
    for i in range(16):
        test=test_data[50*i:50*(i+1),:]
        test_class=0 #分类表示符，默认等于0代表还未进行分类
        for k in range(len(net_sum)):
            time.sleep(0.5)    #方便观察过程，让程序等待0.5s
            e_q=net_sum[k]['net'].quantization_error(test)
            if e_q <= net_sum[k]['max_quantization_error']:
                test_class=k+1
                print('classify into class '+str(k+1))
            else:
                continue
        if test_class==0:   #遇到不在已知工况范围内的情况，学习新的网络进行学习
            print('abnormal case! learning a new net')
            learnt_net = {}
            size = math.ceil(np.sqrt(5 * np.sqrt(len(test))))
            som = MiniSom(size, size, feature_number, sigma=3, learning_rate=0.5,
                      neighborhood_function='bubble')
            som.train_batch(test, max_iter, verbose=True)
            learnt_net['net'] = som
            tmp = []
            for k in range(len(test)):
                tmp.append(som.quantization_error(test[k:k + 1, :]))
            learnt_net['mean_quantization_error'] = np.mean(tmp)  # 计算网络训练时的平均量化误差 （等同于论文中的Eq-training）
            learnt_net['max_quantization_error'] = np.max(tmp)  # 计算网络训练时的最大量化误差 (等同于论文中的d_max-training)
            net_sum.append(learnt_net)
    #print(net_sum[2]['net'].quantization_error(test_data[100:101,:]),net_sum[2]['max_quantization_error'])
    '''
    知乎example
    N=train_data.shape[0]
    M=train_data.shape[1]
    size = math.ceil(np.sqrt(5 * np.sqrt(N))) #决定输出层的经验公式
    max_iter=100

    som = MiniSom(size, size, M, sigma=1.5, learning_rate=0.5,
                  neighborhood_function='bubble')
    som.pca_weights_init(train_data)
    som.train_batch(train_data, max_iter, verbose=False)
    heatmap=som.distance_map()
    active_map=som.activation_response(train_data)
    weight=som.get_weights()
    plt.imshow(heatmap, cmap='Greys',interpolation='nearest')  # miniSom案例中用的pcolor函数,需要调整坐标
    plt.colorbar()
    plt.show()
    '''
    print(0)    #调试专用，用于设置断点