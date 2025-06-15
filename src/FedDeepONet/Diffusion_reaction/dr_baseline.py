import numpy as np
import deepxde as dde
from deepxde.backend import torch
from scipy.io import loadmat
import skopt
from pathos.pools import ProcessPool
from scipy import interpolate
from scipy.integrate import solve_ivp
from sklearn import gaussian_process as gp
from sklearn.preprocessing import StandardScaler
import config
import copy
import random


m = 101
dim_x = 2

loss = []

for i in range(5):

    # Load dataset
    train_data = np.load("dr_traindata200_0.1.npz", allow_pickle=True)
    test_data = np.load("dr_testdata1000_0.1.npz", allow_pickle=True)

    # X_train = (np.repeat(train_data["X_train0"][:1000, :], 10201, axis=0), np.tile(train_data["X_train1"], (1000, 1)))
    # y_train = train_data["y_train"][:1000, :].reshape(-1, 1)
    # X_test = (np.repeat(test_data["X_test0"][:1000, :], 10201, axis=0), np.tile(test_data["X_test1"], (1000, 1)))
    # y_test = test_data["y_test"][:1000, :].reshape(-1, 1)

    # data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    # net = dde.nn.deeponet.DeepONet(
    #     [m, 64, 64], [dim_x, 64, 64],
    #     "relu",
    #     "Glorot normal",
    # )
    X_train, y_train = (train_data["X_train0"].astype(np.float32), train_data["X_train1"].astype(np.float32)), train_data["y_train"].astype(np.float32)
    X_test, y_test = (test_data["X_test0"].astype(np.float32), test_data["X_test1"].astype(np.float32)), test_data["y_test"].astype(np.float32)
    data = dde.data.TripleCartesianProd(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    net = dde.nn.DeepONetCartesianProd(
        [m, 64, 64], [dim_x, 64, 64],
        "relu",
        "Glorot normal",
    )

    # Define a Model
    model = dde.Model(data, net)

    # Compile and Train
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(iterations=15000)
    loss.append(losshistory.metrics_test[-1])

np.savez("dr_baseline.npz", loss=loss)
