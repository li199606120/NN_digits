# encoding: UTF-8
import numpy as np
import cost
import configuration
import glob
import scipy.misc as sc
import scipy.optimize as optimize
import time
import pickle
import datetime

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

# def load_in():
#     model_file="H:/python_operate/digits/data_digits.pkl"
#     with open(model_file,'rb') as str:     #二进制
#         data_value=pickle.load(str)
#         x_values_train=data_value["x_values_train"]
#         y_values_train=data_value["y_values_train"]
#     return x_values_train,y_values_train

def initalize_theta():
    return np.random.randn((configuration.input_layer_size+1)*configuration.hidden1_layer_size+(configuration.hidden1_layer_size+1)*configuration.out_layer_size)

def save(hidden_layer_size, optimized_theta,lamda):
    model = {'hidden_layer_size': hidden_layer_size, 'optimized_theta': optimized_theta,'lamda':lamda}
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    lambda_value_and_hidden_layers = "_hidden" + str(hidden_layer_size)
    timestamp_with_lambda_value = timestamp + lambda_value_and_hidden_layers
    model_filename = "model_" + timestamp_with_lambda_value + ".pkl"
    with open("H:/python_operate/digits/" + model_filename, 'wb') as output_file:
        pickle.dump(model, output_file, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    theta = initalize_theta()
    x_values_train, y_values_train=settle_data()
    apt_theta,apt_cost,apt_dic=optimize.fmin_l_bfgs_b(cost.cost_function,theta,fprime=cost.gradient,args=[x_values_train,y_values_train])
    save(configuration.hidden1_layer_size,apt_theta,configuration.LAMDA)
    print(apt_theta,apt_cost)

    #data_input,data_output=load_in()
    #print len(data_input[0]),len(data_output)
    # print glob.glob("H:/100/*.jpg")   #返回一个list
    # a=[[1,2],[3,4]]
    # b=[[5,6],[7,8]]
    # print np.power(a,2)
    # np.random.seed(0)
    # print np.random.randn(2)
    # t={"1":2,"2":3}
    # o=pickle.dumps(t)
    # print o
    # print pickle.loads(o)
    #f=glob.iglob("H:/100/*.jpg")  #返回一个生成器
    #print [x for x in f]
