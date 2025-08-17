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

class InverseNS:
    def __init__(self, layer_size, activation, initializer, local_epochs, global_epochs, geomtime):
        self.net = dde.nn.FNN(layer_size, activation, initializer)
        self.C1 = dde.Variable(0.0)
        self.C2 = dde.Variable(0.0)
        self.variable = None
        self.le = local_epochs
        self.ge = global_epochs
        self.geomtime = geomtime
        self.data = None
        self.model = None
    def Navier_Stokes_Equation(self, x,y):
        u = y[:, 0:1]
        v = y[:, 1:2]
        p = y[:, 2:3]
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        du_y = dde.grad.jacobian(y, x, i=0, j=1)
        du_t = dde.grad.jacobian(y, x, i=0, j=2)
        dv_x = dde.grad.jacobian(y, x, i=1, j=0)
        dv_y = dde.grad.jacobian(y, x, i=1, j=1)
        dv_t = dde.grad.jacobian(y, x, i=1, j=2)
        dp_x = dde.grad.jacobian(y, x, i=2, j=0)
        dp_y = dde.grad.jacobian(y, x, i=2, j=1)
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
        continuity = du_x + dv_y
        x_momentum = du_t + self.C1 * (u * du_x + v * du_y) + dp_x - self.C2 * (du_xx + du_yy)
        y_momentum = dv_t + self.C1 * (u * dv_x + v * dv_y) + dp_y - self.C2 * (dv_xx + dv_yy)
        return [continuity, x_momentum, y_momentum]
    def set_data(self, bcs):
        self.data = dde.data.TimePDE(
            self.geomtime,
            self.Navier_Stokes_Equation,
            bcs,
            num_domain=700,
            num_boundary=200,
            num_initial=100,
        )
    def set_variable(self, filename):
        self.variable = dde.callbacks.VariableValue([self.C1, self.C2], period=self.le, filename = filename)
    def compile(self):
        self.model = dde.Model(self.data, self.net)
        self.model.compile("adam", lr=1e-3, external_trainable_variables=[self.C1, self.C2], decay = ('step', self.ge*self.le/2, 0.1))
    def train(self):
        self.model.train(iterations=self.le, callbacks=[self.variable], disregard_previous_best=True)
    def get_loss_history(self):
        return np.array(self.model.losshistory.loss_train)
        
# Load training data
def load_training_data(num,x1,x2,y1,y2):
    data = loadmat("cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2
    N = X_star.shape[0]
    T = t_star.shape[0]
    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 1]
    data1 = np.concatenate([x, y, t, u, v, p], 1)
    data2 = data1[:, :][data1[:, 2] <= 1] ### T=1
    data3 = data2[:, :][data2[:, 0] >= x1]
    data4 = data3[:, :][data3[:, 0] <= x2]
    data5 = data4[:, :][data4[:, 1] >= y1]
    data_domain = data5[:, :][data5[:, 1] <= y2]
    # choose number of training points: num =7000
    idx = np.random.choice(data_domain.shape[0], num, replace=False)
    x_train = data_domain[idx, 0:1]
    y_train = data_domain[idx, 1:2]
    t_train = data_domain[idx, 2:3]
    u_train = data_domain[idx, 3:4]
    v_train = data_domain[idx, 4:5]
    p_train = data_domain[idx, 5:6]
    return [x_train, y_train, t_train, u_train, v_train, p_train]

def generate_block(n_clients, piece, geom, is_split_x):
    if isinstance(geom, dde.geometry.Rectangle):
        if is_split_x:
            geo_interval = (geom.xmax[0] - geom.xmin[0]) / n_clients / piece
        else:
            geo_interval = (geom.xmax[1] - geom.xmin[1]) / n_clients / piece
        drift = geo_interval * n_clients
        geo_list = []
        for j in range(n_clients):
            client_area_list = []
            for i in range(piece):
                if is_split_x:
                    geo_start = (geom.xmin[0] + i*drift+j*geo_interval, geom.xmin[1])
                    geo_end = (geom.xmin[0] + i*drift + (j+1)*geo_interval, geom.xmax[1])
                else:
                    geo_start = (geom.xmin[0], geom.xmin[1] + i*drift+j*geo_interval)
                    geo_end = (geom.xmax[0], geom.xmin[1] + i*drift + (j+1)*geo_interval)
                client_area_list.append((geo_start, geo_end))
            geo_list.append(client_area_list)
        return geo_list
    
def client_data_generator(geom, n_clients, piece, is_split_x):
    geo_list = generate_block(n_clients, piece, geom, is_split_x)
    data = []
    observe_u = []
    observe_v = []
    for i in range(n_clients):
        client_data = []
        ob_u_list = []
        ob_v_list = []
        cur_list = geo_list[i]
        for tuple in cur_list:
            [ob_x, ob_y, ob_t, ob_u, ob_v, ob_p] = load_training_data(num=int(700/n_clients/piece), x1=tuple[0][0], x2=tuple[1][0], y1=tuple[0][1], y2=tuple[1][1])
            ob_xyt = np.hstack((ob_x, ob_y, ob_t))
            ob_u_list.append(ob_u)
            ob_v_list.append(ob_v)
            client_data.append(ob_xyt)
        client_data = np.concatenate(client_data, axis=0)
        all_ob_u = np.concatenate(ob_u_list, axis=0)
        all_ob_v = np.concatenate(ob_v_list, axis=0)
        data.append(client_data)
        observe_u.append(dde.icbc.PointSetBC(client_data, all_ob_u, component=0))
        observe_v.append(dde.icbc.PointSetBC(client_data, all_ob_v, component=1))
    return data, observe_u, observe_v