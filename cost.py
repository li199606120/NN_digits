# coding=utf-8
import numpy as np
import activate
import configuration


def cost_function(theta, x_values, y_values):
    theta_input2hidden1 = theta[0:(configuration.input_layer_size + 1) * configuration.hidden1_layer_size]
    theta_hidden12out = theta[(configuration.input_layer_size + 1) * configuration.hidden1_layer_size:]

    theta1 = theta_input2hidden1.reshape(configuration.input_layer_size + 1,
                                         configuration.hidden1_layer_size)  # (input+1)*hidden
    theta2 = theta_hidden12out.reshape(configuration.hidden1_layer_size + 1,
                                       configuration.out_layer_size)  # (hidden+1)*out
    number_input = x_values.shape[0]

    hidden1_input = np.c_[np.ones(number_input), x_values].dot(theta1)  # m*hidden
    hidden1_output = activate.sigmoid(hidden1_input)  # m*hidden

    hidden2_input = np.c_[np.ones(number_input), hidden1_output].dot(theta2)  # m*out
    out = activate.sigmoid(hidden2_input)  # m*out

    sum_theta = np.append(theta1.flatten(), theta2.flatten())  # append必须加，两个向量合并成一个
    regularization_term = configuration.LAMDA / 2.0 / number_input * np.sum(np.power(sum_theta, 2))
    cost_J = np.sum(np.sum(np.power(out - y_values, 2))) * 1.0 / number_input
    cost_J = cost_J + regularization_term

    return cost_J


def gradient(theta, x_values, y_values):
    theta_input2hidden1 = theta[0:(configuration.input_layer_size + 1) * configuration.hidden1_layer_size]
    theta_hidden12out = theta[(configuration.input_layer_size + 1) * configuration.hidden1_layer_size:]

    theta1 = theta_input2hidden1.reshape(configuration.input_layer_size + 1,
                                         configuration.hidden1_layer_size)  # (input+1)*hidden
    theta2 = theta_hidden12out.reshape(configuration.hidden1_layer_size + 1,
                                       configuration.out_layer_size)  # (hidden+1)*out

    number_input = x_values.shape[0]

    hidden1_input = np.c_[np.ones(number_input), x_values].dot(
        theta1)  # 全1项放在了最左边   m*(input+1) (input+1)*hidden = m*hidden
    hidden1_output = activate.sigmoid(hidden1_input)  # 维度不变

    hidden2_input = np.c_[np.ones(number_input), hidden1_output].dot(theta2)  # m*(hidden+1) (hidden+1)*out = m*out
    out = activate.sigmoid(hidden2_input)

    error = out - y_values  # m*out   最后一层激活函数为y=x
    backpropagated_errors = error.dot(theta2.T[:, 1:]) * activate.sigmoid_gradient(hidden1_input)  # m*hidden
    delta_1 = np.c_[np.ones(number_input), x_values].T.dot(backpropagated_errors)  # (input+1)*hidden
    delta_2 = np.c_[np.ones(hidden1_output.shape[0]), hidden1_output].T.dot(error)  # (hidden+1)*out

    theta1_gradient = 1.0 / number_input * delta_1 + configuration.LAMDA / number_input * theta1  # (input+1)*hidden
    theta2_gradient = 1.0 / number_input * delta_2 + configuration.LAMDA / number_input * theta2

    theta = np.append(theta1_gradient.flatten(), theta2_gradient.flatten())

    return theta








