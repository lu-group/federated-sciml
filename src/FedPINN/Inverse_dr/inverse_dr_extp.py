"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle
Implementation for the diffusion-reaction system with a space-dependent reaction rate in paper https://arxiv.org/abs/2111.02801.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from scipy.integrate import solve_bvp
import deepxde as dde
import copy
from deepxde.backend import torch

l = 0.01


def k(x):
    return 0.1 + np.exp(-0.5 * (x - 0.5) ** 2 / 0.15**2)


def fun(x, y):
    return np.vstack((y[1], (k(x) * y[0] + np.sin(2 * np.pi * x)) / l))


def bc(ya, yb):
    return np.array([ya[0], yb[0]])

a = np.linspace(0, 1, 1000)
b = np.zeros((2, a.size))

res = solve_bvp(fun, bc, a, b)


def sol(x):
    return res.sol(x)[0]


def du(x):
    return res.sol(x)[1]


def gen_traindata(num):
    xvals = np.linspace(0, 1, num)
    yvals = sol(xvals)

    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))


def output_transform(x, y):
    return torch.cat(
        (torch.tanh(x) * torch.tanh(1 - x) * y[:, 0:1], y[:, 1:2]), dim=1
    )


def pde(x, y):
    u, k = y[:, 0:1], y[:, 1:2]
    du_xx = dde.grad.hessian(y, x, component=0)
    return l * du_xx - u * k - dde.backend.sin(2 * np.pi * x)


# def func(x):
#     return 0


def data_assignmentX_1d(X_train, piece, n_clients):

    datalen = len(X_train)
    quotient =  datalen // (piece * n_clients)
    residue = datalen % (piece * n_clients)

    data_listx = []

    for j in range(n_clients):
        tempx = []
        resx = []

        for i in range(piece):
            tempx.append(X_train[(i*n_clients+j)*quotient:(i*n_clients+j)*quotient+quotient])

        extra = 0
        while extra * n_clients + j < residue:
            resx.append(X_train[quotient*piece*n_clients+ j + extra * n_clients])
            extra +=1

        data_listx.append(np.vstack((np.array(tempx).reshape(-1,1), np.array(resx).reshape(-1,1))))

    return data_listx


# data generation
num = 24
geom = dde.geometry.Interval(0, 1)
ob_x, ob_u = gen_traindata(num)
bc = dde.icbc.DirichletBC(geom, sol, lambda _, on_boundary: on_boundary, component=0)
observe_u = dde.PointSetBC(ob_x, ob_u, component=0)

n_pieces =  [1, 2, 3, 4, 6, 10]
n_clients = 2
global_epochs = 20000
local_epochs = 5
x = np.linspace(0, 1, 1001)[:, None]

def l2(a, b):
    return dde.metrics.l2_relative_error(a, b)

for iter in range(5):   
    k_loss1 = []
    k_loss2 = []
    u_loss1 = []
    u_loss2 = []
    
    for piece in n_pieces:
        subdata_x = data_assignmentX_1d(ob_x, piece=piece, n_clients=n_clients)
        subdata_u = data_assignmentX_1d(ob_u, piece=piece, n_clients=n_clients)

        # Client models
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", "Glorot uniform"))
            observe_u = dde.icbc.PointSetBC(subdata_x[i], subdata_u[i], component=0)
            client_nets[i].apply_output_transform(output_transform)
            client_data = dde.data.PDE(geom, pde, bcs=[observe_u], num_domain=8, num_boundary=2,
                                        train_distribution="uniform", num_test=1000)
            model = dde.Model(client_data, client_nets[i])
            client_models.append(model)
            client_models[i].compile("adam", lr=1e-4, metrics = [])


        losshistory1, train_state = client_models[0].train(iterations = local_epochs*global_epochs)
        losshistory2, train_state = client_models[1].train(iterations = local_epochs*global_epochs)

        # x = geom.uniform_points(500)
    
        ktrue = k(x)
        utrue = res.sol(x)[0]

        yhat1 = client_models[0].predict(x)
        uhat1, khat1 = yhat1[:, 0:1], yhat1[:, 1:2]
        yhat2 = client_models[1].predict(x)
        uhat2, khat2 = yhat2[:, 0:1], yhat2[:, 1:2]

        k_loss1.append(dde.metrics.l2_relative_error(khat1, ktrue))
        k_loss2.append(dde.metrics.l2_relative_error(khat2, ktrue))
        u_loss1.append(dde.metrics.l2_relative_error(uhat1, utrue))
        u_loss2.append(dde.metrics.l2_relative_error(uhat2, utrue))

       
    np.savez("inverse_dr_extp.npz".format(iter), k_loss1=k_loss1, k_loss2=k_loss2, u_loss1=u_loss1, u_loss2=u_loss2)
