import numpy as np
import deepxde as dde
from deepxde.backend import torch
from scipy.io import loadmat
import skopt
from pathos.pools import ProcessPool
from scipy import interpolate
from scipy.integrate import solve_ivp
from sklearn import gaussian_process as gp
# from spaces import GRF
import config
import copy
import random


m = 100
dim_x = 1
num = 100

loss = []

X_train0 = np.load("traindata500_0.1.npz", allow_pickle=True)['X_train0']
X_train1 = np.load("traindata500_0.1.npz", allow_pickle=True)['X_train1']
y_train = np.load("traindata500_0.1.npz", allow_pickle=True)['y_train']
X_test0 = np.load("testdata1000_0.1.npz", allow_pickle=True)['X_test0']
X_test1 = np.load("testdata1000_0.1.npz", allow_pickle=True)['X_test1']
y_test = np.load("testdata1000_0.1.npz", allow_pickle=True)['y_test']

X_train = (X_train0.astype(np.float32), X_train1.astype(np.float32))
y_train = y_train.astype(np.float32)
X_test= (X_test0.astype(np.float32), X_test1.astype(np.float32))
y_test = y_test.astype(np.float32)

for i in range(5):
    data = dde.data.TripleCartesianProd(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    net = dde.nn.DeepONetCartesianProd(
        [m, 40, 40], [dim_x, 40, 40],
        "relu",
        "Glorot normal",
    )
    # Define a Model
    model = dde.Model(data, net)

    # Compile and Train
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(iterations=15000)
    loss.append(losshistory.metrics_test[-1])

np.savez("antiderivative_baseline.npz", loss=loss)