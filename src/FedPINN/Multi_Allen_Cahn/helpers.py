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
import skopt


def gen_testdata():
    data = loadmat("Allen_Cahn.mat")

    t = data["t"]
    x = data["x"]
    u = data["u"]

    dt = dx = 0.01
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y

class Multi_Allen_Cahn:
    def __init__(self, layer_size, activation, initializer, local_epochs, global_epochs, geomtime):
        self.net = dde.nn.FNN(layer_size, activation, initializer)
        self.le = local_epochs
        self.ge = global_epochs
        self.geomtime = geomtime
        self.d = 0.001
        self.data = None
        self.model = None
    def pde(self, x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - self.d * dy_xx - 5 * (y - y**3)
    def func(self, x):
        sol = x + 1 / 8 * np.sin(8 * x)
        for i in range(1, 5):
            sol += 1 / i * np.sin(i * x)
        return sol
    def transform(self, x, y):
        return x[:, 0:1]**2 * torch.cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1]**2) * y
    def set_data(self, anchor):
        self.data = dde.data.TimePDE(
            self.geomtime, 
            self.pde, 
            [], 
            anchors= anchor)
        
    def compile(self):
        self.net.apply_output_transform(self.transform)
        self.model = dde.Model(self.data, self.net)
        self.model.compile("adam", lr=1e-3)
    def train(self):
        self.model.train(iterations=self.le)
    def get_loss_history(self):
        return np.array(self.model.losshistory.loss_train)
    def get_metric_history(self):
        return np.array(self.model.losshistory.metrics_test)
    def return_l2_error(self, x, y):
        y_pred = self.model.predict(x)
        return np.linalg.norm(y_pred - y,ord=2) / np.linalg.norm(y,ord=2)
    
def point_generator(block_tuple, num_points):
    generator = skopt.sampler.Hammersly()
    xmin = block_tuple[0]
    xmax = block_tuple[1]
    xx = generator.generate([(xmin[0], xmax[0]),(xmin[1], xmax[1])], num_points)
    return np.array(xx)

def generate_geometry_interval(n_clients, piece, geotime):
    if isinstance(geotime, dde.geometry.GeometryXTime):
        geomin = geotime.geometry.l
        geomax = geotime.geometry.r
        geo_interval = (geomax - geomin) / n_clients / piece
        drift = geo_interval * n_clients
        geo_list = []
        for j in range(n_clients):
            client_area_list = []
            for i in range(piece):
                geotime_start = (geomin + i*drift+j*geo_interval, geotime.timedomain.t0)
                geotime_end = (geomin + i*drift + (j+1)*geo_interval, geotime.timedomain.t1)
                client_area_list.append((geotime_start, geotime_end))
            geo_list.append(client_area_list)
        return geo_list

def data_generation_Hammersly(piece, ntrain, n_clients, interval):
    '''
    ntrain: num of sample points in total per axis
    piece: the number of cut on each axis
    interval should be the form ((0, 0), (1, 1))
    This method for now only supports 2 clients
    '''
    num_points = int(ntrain / n_clients / piece)
    subblocks = generate_geometry_interval(n_clients, piece, interval)
    sample_data = []
    for block in subblocks:
        client_data = []
        for i in range(piece):
            client_data.append(point_generator(block[i],num_points=num_points))
        sample_data.append(np.concatenate(client_data, axis = 0))
    return sample_data