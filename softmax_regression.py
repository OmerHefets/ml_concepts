import numpy as np


def add_bias(data):
    data = np.c_[np.ones((data.shape[0], 1)), data]
    return data


def s_k_vector(theta_matrix, x_vector):
    return theta_matrix.T.dot(x_vector)


def softmax_func_vector(sk_vector):
    exp_sk = np.exp(sk_vector)
    sum_exp_sk = sum(exp_sk)
    softmax_prob_vector = exp_sk / sum_exp_sk
    return softmax_prob_vector


# applies when classes comes in numbers
def y_one_hot_matrix(y, k):
    # matrix shape = [m * k]
    y_matrix = np.ones((y.shape[0], k))
    for i in range(k):
        y_matrix[:, i:i+1] = (y == (i+1)) # first class == 1
    return y_matrix

