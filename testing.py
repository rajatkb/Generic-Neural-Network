import numpy as np
import math
from sklearn import datasets
from generic_nn import generic_nn as gnn


def accuracy(Y,Yn):
    C = (np.array(Y) == np.array(Yn))
    return np.count_nonzero(C)/C.shape[1]

diabetes = datasets.load_breast_cancer()
data = diabetes["data"].T
target = diabetes["target"].reshape(1,len(diabetes["target"]))


train_percent=0.8

X_train = data[:,0:math.floor(data.shape[1]*train_percent)]
Y_train = target[:,0:math.floor(data.shape[1]*train_percent)]

print("Train data:" , X_train)
print("Train result:", Y_train)

          
neural_net = gnn()



neural_net.output_layer(1 , "sigmoid")

neural_net.train(X_train , Y_train ,
                 epoch=1000000 , alpha=0.001 ,
                 drop_keep=0.9 , batches=1 , adam=[0.5,0.999,10e-8], l2=0.9 , show_graph=True)

Yn = neural_net.predict(X_train)
Yn = (Yn > 0.5)
print("Accuracy: " , accuracy(Y_train , Yn))
