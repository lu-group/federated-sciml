import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import re
import mlnotify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import argparse
from pathlib import Path


class OneDimPoisson:
    def __init__(self, layer_size, activation, initializer, local_epochs, global_epochs, geom):
        self.net = dde.nn.FNN(layer_size, activation, initializer)
        self.le = local_epochs
        self.ge = global_epochs
        self.geom = geom
        self.data = None
        self.model = None
    def Poisson(self, x,y):
        dy_xx = dde.grad.hessian(y, x)
        f = 8 * torch.sin(8 * x)
        for i in range(1, 5):
            f += i * torch.sin(i * x)
        return -dy_xx - f
    def func(self, x):
        sol = x + 1 / 8 * np.sin(8 * x)
        for i in range(1, 5):
            sol += 1 / i * np.sin(i * x)
        return sol
    def transform(self, x, y):
        return x + torch.tanh(x) * torch.tanh(np.pi - x) * y
    def set_data(self, anchor):
        self.data = dde.data.PDE(
            self.geom, 
            self.Poisson, 
            [], 
            anchors= anchor.reshape(len(anchor),1), 
            solution= self.func, 
            num_test = 100)
    def compile(self):
        self.net.apply_output_transform(self.transform)
        self.model = dde.Model(self.data, self.net)
        self.model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    def train(self):
        self.model.train(iterations=self.le)
    def get_loss_history(self):
        return np.array(self.model.losshistory.loss_train)
    def get_metric_history(self):
        return np.array(self.model.losshistory.metrics_test)
    def return_l2_error(self, x, y):
        y_pred = self.model.predict(x)
        return np.linalg.norm(y_pred - y,ord=2) / np.linalg.norm(y,ord=2)
    
def data_assignment(n_client, x,y, piece):
    assert len(x) == len(y)
    datalen= len(x)
    quotient =  datalen // (piece * n_client)
    residue = datalen % (piece * n_client)
    ### each client will get quotient * piece data if ignoring the residue
    x_list = []
    y_list = []
    for j in range(n_client):
        tempx = []
        tempy = []
        for i in range(piece):
            tempx += list(x[(i*n_client+j)*quotient:(i*n_client+j)*quotient+quotient])
            tempy +=list(y[(i*n_client+j)*quotient:(i*n_client+j)*quotient+quotient])
        extra = 0
        while extra * n_client + j < residue:
            tempx.append(x[quotient*piece*n_client+ j + extra * n_client])
            tempy.append(y[quotient*piece*n_client+ j + extra * n_client])
            extra +=1
        tempx = np.resize(np.array(tempx, dtype = float), (int(datalen / n_client),1))
        tempy = np.resize(np.array(tempy, dtype = float), (int(datalen / n_client),1))
        x_list.append(tempx)
        y_list.append(tempy)
    return x_list, y_list