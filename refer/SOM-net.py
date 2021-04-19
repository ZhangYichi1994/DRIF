from minisom import MiniSom
import math
import numpy as np
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

'''
函数train_split
功能：将原始的训练数据按照已知的类别个数进行相应的划分
'''
def train_split(train_data,element_in_cluster):
    result=[]
    tmp=0
    for i in range(len(element_in_cluster)):
        result.append(train_data[tmp:tmp+element_in_cluster[i],:])
        tmp=tmp+element_in_cluster[i]
    return result


if __name__ == '__main__':
    '''
    CSTH数据集
    train_data = loadmat('origin-data')['data']
    test_data=train_data
    '''
    train_data = loadmat('origin-data')['data']
    test_data=train_data
    '''
    UCI数据集
    fname = "Iris.txt"
    origin_data=np.loadtxt(fname, delimiter=",")
    train_data=origin_data[:,1:]
    test_data=origin_data[:,1:]
    '''
    '''
    训练数据与测试数据标准化
    '''
    train_mean=np.mean(train_data,axis=0)
    train_std=np.std(train_data,axis=0)
    train_data=(train_data-train_mean)/train_std

    test_mean=np.mean(test_data,axis=0)
    test_std=np.std(test_data,axis=0)
    test_data=(test_data-test_mean)/test_std

    #参数设置
    net_sum=[]  #集合所有已经训练好的SOM网络
    element_in_cluster=[200,200]  #每个类别中元素的个数
    feature_number=train_data.shape[1]  #训练数据的特征个数
    train_data_split=train_split(train_data,element_in_cluster)
    max_iter=300
    #网络训练阶段
    for i in range(len(train_data_split)):
        learnt_net={}
        size = math.ceil(np.sqrt(5 * np.sqrt(element_in_cluster[i])))
        som = MiniSom(size, size, feature_number, sigma=2, learning_rate=0.5,
                      neighborhood_function='bubble')
        som.train_batch(train_data_split[i],max_iter,verbose=True)
        learnt_net['net']=som
        tmp=[]
        for k in range(len(train_data_split[i])):
            tmp.append(som.quantization_error(train_data_split[i][k:k+1,:]))
        print(tmp)
        learnt_net['mean_quantization_error']=np.mean(tmp)  #计算网络训练时的平均量化误差 （等同于论文中的Eq-training）
        learnt_net['max_quantization_error']=np.max(tmp)    #计算网络训练时的最大量化误差 (等同于论文中的d_max-training)
        net_sum.append(learnt_net)

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