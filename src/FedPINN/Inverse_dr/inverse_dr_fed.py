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

# n_pieces =  [1, 2, 3, 4, 6, 10]
n_pieces = [1]
n_clients = 2
global_epochs = 20000
local_epochs = 5
x = np.linspace(0, 1, 1001)[:, None]

for iter in range(5):   
    k_loss = []
    u_loss = []

    wd_fc11 = []
    wd_fc12 = []
    wd_fc13 = []
    wd_fc14 = []
    wd_fc21 = []
    wd_fc22 = []
    wd_fc23 = []
    wd_fc24 = []

    for piece in n_pieces:
        subdata_x = data_assignmentX_1d(ob_x, piece=piece, n_clients=n_clients)
        subdata_u = data_assignmentX_1d(ob_u, piece=piece, n_clients=n_clients)

        # Central model
        central_net = dde.maps.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", "Glorot uniform")
        central_net.apply_output_transform(output_transform)
        central_data = dde.data.PDE(geom,pde,bcs=[observe_u],num_domain=8,num_boundary=2,
                                           train_distribution="uniform",num_test=1000)
        Central = dde.Model(central_data, central_net)
        Central.compile("adam", lr=1e-4, metrics=[])

        # Server model
        server_net = dde.nn.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", "Glorot uniform")
        server_net.apply_output_transform(output_transform)
        data = dde.data.PDE(geom, pde, bcs=[observe_u], num_domain=8, num_boundary=2, 
                                   train_distribution="uniform", num_test=1000)
        server_model = dde.Model(data, server_net)
        server_model.compile("adam", lr=1e-4, metrics = [])

        # Client models
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", "Glorot uniform"))
            client_nets[i].apply_output_transform(output_transform)
            observe_u = dde.icbc.PointSetBC(subdata_x[i], subdata_u[i], component=0)
            client_data = dde.data.PDE(geom, pde, bcs=[observe_u], num_domain=8, num_boundary=2,
                                        train_distribution="uniform", num_test=1000)
            model = dde.Model(client_data, client_nets[i])
            client_models.append(model)
            client_models[i].compile("adam", lr=1e-4, metrics = [])

        # train
        k_l2 = []
        u_l2 = []
        fc1 = []
        fc2 = []
        fc3 = []
        fc4 = []
        fc5 = []
        fc6 = []
        fc7 = []
        fc8 = []

        for i in range(global_epochs):
            Central.train(iterations = local_epochs)

            # Train the federated models
            for j in range(len(client_models)):
                train_state = client_models[j].train(iterations=local_epochs) 

            # aggregate
            combined_dict = copy.deepcopy(client_models[0].state_dict())
            for (k1,v1), (k2,v2) in zip(client_models[0].state_dict().items(), client_models[1].state_dict().items()):
                combined_dict[k1] = (v1+v2)/2

            # broadcast
            for j in range(len(client_models)):
                client_nets[j].load_state_dict(combined_dict)
            
            # (optimal) test and calculate error
            if i==0 or (i+1)%1000 == 0:
                server_net.load_state_dict(combined_dict)

                yhat = server_model.predict(x)
                uhat, khat = yhat[:, 0:1], yhat[:, 1:2]

                # test for k
                ktrue = k(x)
                k_l2.append(dde.metrics.l2_relative_error(khat, ktrue))

                # test for u
                utrue = res.sol(x)[0]
                u_l2.append(dde.metrics.l2_relative_error(uhat, utrue))
                
                WSGD = copy.deepcopy(Central.net.state_dict())
                Wfedavg = copy.deepcopy(server_model.net.state_dict())
            
                fc1.append(np.linalg.norm(Wfedavg['layers.0.0.weight'].cpu().numpy() - WSGD['layers.0.0.weight'].cpu().numpy()) / np.linalg.norm(WSGD['layers.0.0.weight'].cpu().numpy()))
                fc2.append(np.linalg.norm(Wfedavg['layers.1.0.weight'].cpu().numpy() - WSGD['layers.1.0.weight'].cpu().numpy()) / np.linalg.norm(WSGD['layers.1.0.weight'].cpu().numpy()))
                fc3.append(np.linalg.norm(Wfedavg['layers.2.0.weight'].cpu().numpy() - WSGD['layers.2.0.weight'].cpu().numpy()) / np.linalg.norm(WSGD['layers.2.0.weight'].cpu().numpy()))
                fc4.append(np.linalg.norm(Wfedavg['layers.3.0.weight'].cpu().numpy() - WSGD['layers.3.0.weight'].cpu().numpy()) / np.linalg.norm(WSGD['layers.3.0.weight'].cpu().numpy()))
                fc5.append(np.linalg.norm(Wfedavg['layers.0.1.weight'].cpu().numpy() - WSGD['layers.0.1.weight'].cpu().numpy()) / np.linalg.norm(WSGD['layers.0.1.weight'].cpu().numpy()))
                fc6.append(np.linalg.norm(Wfedavg['layers.1.1.weight'].cpu().numpy() - WSGD['layers.1.1.weight'].cpu().numpy()) / np.linalg.norm(WSGD['layers.1.1.weight'].cpu().numpy()))
                fc7.append(np.linalg.norm(Wfedavg['layers.2.1.weight'].cpu().numpy() - WSGD['layers.2.1.weight'].cpu().numpy()) / np.linalg.norm(WSGD['layers.2.1.weight'].cpu().numpy()))
                fc8.append(np.linalg.norm(Wfedavg['layers.3.1.weight'].cpu().numpy() - WSGD['layers.3.1.weight'].cpu().numpy()) / np.linalg.norm(WSGD['layers.3.1.weight'].cpu().numpy()))

        k_loss.append(k_l2)
        u_loss.append(u_l2)

        wd_fc11.append(fc1)
        wd_fc12.append(fc2)
        wd_fc13.append(fc3)
        wd_fc14.append(fc4)
        wd_fc21.append(fc5)
        wd_fc22.append(fc6)
        wd_fc23.append(fc7)
        wd_fc24.append(fc8)

    print("l2 relative error for k: " + str(k_l2))
    print("l2 relative error for u: " + str(u_l2))
    np.savez("inverse_dr_fed{}.npz".format(iter), k_loss=k_loss, u_loss=u_loss)
    np.savez("inverse_dr_fed{}_wd.npz".format(iter), fc1 = wd_fc11, fc2 = wd_fc12, fc3 = wd_fc13, fc4 = wd_fc14,
             fc5 = wd_fc21, fc6 = wd_fc22, fc7 = wd_fc23, fc8 = wd_fc24)
