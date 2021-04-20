import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde as kde
from scipy.optimize import fsolve
from scipy import integrate as inter

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
'''
测试备用
def f(x):
    return density(x)
def f2(x):
    return inter.quad(f,-np.inf,x)[0]-confidence
'''

if __name__ == '__main__':
    data = np.array(pd.read_csv('kde.csv', header=None)).T #读取用于测试的数据
    cal_limit=kde_limit(0.99,data[0])   #创建一个kde控制限计算的对象（创建时要输入两个参数：1.置信度 2.输入数据）
    limit2=cal_limit.fit()              #调用该对象中计算功能（无需输入其他参数，自动进行计算）
    print(limit2)

    '''
    confidence = 0.99
    density = kde(y)
    limit=fsolve(f2,(y.max()-y.min())/2)
    '''



