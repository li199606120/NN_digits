#coding=utf-8
import glob
import numpy as np
import pickle
import sys

def settle_data():
    txt_name = glob.glob("H:/python_operate/digits/trainingDigits/*.txt")
    x_values_train = []
    y_values_train=[]
    for filename in txt_name:
        txt_file = open(filename, 'r')
        buf = txt_file.read()
        temp = []
        for i in buf:
            if i != '\n':
                temp.append(int(i))
        x_values_train.append(temp)
        txt_file.close()
        #处理y数据
        temp=filename.split("\\")
        y_values_train.append([int(temp[1][0])])
    #将y_values转换为01矩阵
    temp=np.zeros([len(x_values_train),10])
    for i in range(len(x_values_train)):
        temp[i][y_values_train[i]]=1
    x_values_train = np.array(x_values_train)
    y_values_train=temp
    return x_values_train,y_values_train


def save(x_values_train,y_values_train):
    model={"x_values_train":x_values_train,"y_values_train":y_values_train}
    with open("H:/python_operate/digits/data_digits.pkl", 'wb') as output_file:     #以binary（二进制）方式打开，写入内容
        pickle.dump(model, output_file, pickle.HIGHEST_PROTOCOL)   #将model写入output_file中

if __name__ == '__main__':
    x_values_train, y_values_train=settle_data()
    save(x_values_train,y_values_train)

