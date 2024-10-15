# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test 

import os
import pickle

def init_network():
    # 현재 스크립트의 절대 경로 구하기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # sample_weight.pkl 파일의 절대 경로 설정
    file_path = os.path.join(current_dir, "sample_weight.pkl")
    
    # 파일 열기
    with open(file_path, 'rb') as f:
        network = pickle.load(f)
    return network



def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0

import numpy as np

# 배치 단위로 분류
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

# 사용된 노드의 수와 층의 수를 print
print("Number of nodes: " + str(len(x)))
print("Number of layers: " + str(len(network)))
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
