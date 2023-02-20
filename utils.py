import numpy as np


# 对tensor转numpy求最大值
def softmax(x):
    x = x[0].numpy()
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    print("摔倒概率为:", y[1])


