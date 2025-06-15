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
from deepxde.backend import torch

l = 0.01


def k(x):
    return 1.0 + np.exp(-0.5 * (x - 0.5) ** 2 / 1.05**2)


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
    # du_xx = dde.grad.hessian(y, x, i=0, j=0, component=0)
    du_xx = dde.grad.hessian(y, x, component=0)
    return l * du_xx - u * k - dde.backend.sin(2 * np.pi * x)

# data generation
num = 24
geom = dde.geometry.Interval(0, 1)
ob_x, ob_u = gen_traindata(num)
bc = dde.icbc.DirichletBC(geom, sol, lambda _, on_boundary: on_boundary, component=0)
observe_u = dde.PointSetBC(ob_x, ob_u, component=0)

global_epochs = 20000
local_epochs = 5
x = np.linspace(0, 1, 1001)[:, None]
k_err = []
u_err = []

for iter in range(5):   

    # Central model
    central_net = dde.maps.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", "Glorot uniform")
    central_net.apply_output_transform(output_transform)
    central_data = dde.data.PDE(geom,pde,bcs=[observe_u],num_domain=8,num_boundary=2,
                                        train_distribution="uniform",num_test=1000)
    Central = dde.Model(central_data, central_net)
    Central.compile("adam", lr=1e-4, metrics=[])
    Central.train(iterations=local_epochs*global_epochs)

    yhat = Central.predict(x)
    uhat, khat = yhat[:, 0:1], yhat[:, 1:2]

    # test for k
    ktrue = k(x)
    k_err.append(dde.metrics.l2_relative_error(khat, ktrue))

    # test for u
    utrue = res.sol(x)[0]
    u_err.append(dde.metrics.l2_relative_error(uhat, utrue))

np.savez("inverse_dr.npz", k_err=k_err, u_err=u_err)