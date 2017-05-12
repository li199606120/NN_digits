#coding=utf-8
import numpy as np
import sympy as sym
def sigmoid(x):
    return 1.0/(1.0+np.power(np.e,-x))

def tanh(x):
    return (np.power(np.e,x)-np.power(np.e,-x))*1.0/(np.power(np.e,x)+np.power(np.e,-x))

def ReLu(x):   #这里x为实数
    return np.max([0,x])

def sigmoid_gradient(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh_gradient(x):
    return 1-(tanh(x))**2

def ReLu_gradient(x):
    return 0 if x<0 else 1

# if __name__ == '__main__':
#     x=[1,3]
#     x=np.array(x)
#     y=np.array([[2,1],[4,3]])
#     print x*y  #必须是数组形式array