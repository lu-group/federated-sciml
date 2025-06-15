import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Any, Dict, List
import copy
import random
import deepxde as dde

def data_assignment_1d(X_train, y_train, piece, n_clients):

    datalen = len(y_train)
    quotient =  datalen // (piece * n_clients)
    residue = datalen % (piece * n_clients)

    data_listx = []
    data_listy = []

    for j in range(n_clients):
        tempx = []
        tempy = []
        resx = []
        resy = []

        for i in range(piece):
            tempx.append(X_train[(i*n_clients+j)*quotient:(i*n_clients+j)*quotient+quotient])
            tempy.append(y_train[(i*n_clients+j)*quotient:(i*n_clients+j)*quotient+quotient])

        extra = 0
        while extra * n_clients + j < residue:
            resx.append(X_train[quotient*piece*n_clients+ j + extra * n_clients])
            resy.append(y_train[quotient*piece*n_clients+ j + extra * n_clients])
            extra +=1

        data_listx.append(np.vstack((np.array(tempx).reshape(-1,1), np.array(resx).reshape(-1,1))))
        data_listy.append(np.vstack((np.array(tempy).reshape(-1,1), np.array(resy).reshape(-1,1))))

    return data_listx, data_listy

# Load dataset

def f(x):
    return (x + 0.5)**4 - np.sin(10 * np.pi * x) / (2*x+3)

ftrain = np.loadtxt("Gramacy&Lee(2012)_train.txt")
ftest = np.loadtxt("Gramacy&Lee(2012)_test.txt")
X_train = ftrain[:,:1]
y_train =ftrain[:,1][:,None]
X_test = ftest[:,:1]
y_test =ftest[:,1][:,None]

# define hyperparameters
n_clients = 2
local_epochs = 3
global_epochs = 5000
layer_size = [1] + [64] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
n_pieces = list(range(1,13))+[14,16,20,25,33,50,100]

# 1D federated
for iter in range(5):

    loss = []
    wd_fc1 = []
    wd_fc2 = []
    wd_fc3 = []
    wd_fc4 = []
    for piece in n_pieces:
        # split data
        subdata_x, subdata_y = data_assignment_1d(X_train, y_train, piece=piece, n_clients=n_clients)

        # directly define server as a FNN
        server_net = dde.nn.FNN(
            layer_size,
            activation,
            initializer,
        )
        data = dde.data.DataSet(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        server_model = dde.Model(data, server_net)
        server_model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
        
        # Define a centralized training comparison group
        central_net = dde.nn.FNN(
            layer_size,
            activation,
            initializer,
        )
        central_data = dde.data.Function(dde.geometry.Interval(-1, 1), f, 200,1000)
        # central_data = dde.data.DataSet(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        Central = dde.Model(central_data, central_net)
        Central.compile("adam", lr=1e-3, metrics=["l2 relative error"])

        # define a list of clients as FNNs with same architecture as server
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.FNN(layer_size,activation,initializer))
            client_data = dde.data.DataSet(X_train=subdata_x[i], y_train=subdata_y[i], X_test=X_test, y_test=y_test)
            model = dde.Model(client_data, client_nets[i])
            client_models.append(model)
            client_models[i].compile("adam", lr=1e-3, metrics=["l2 relative error"])

        # train
        l2 = []
        fc1 = []
        fc2 = []
        fc3 = []
        fc4 = []
        for i in range(global_epochs):
            ### Train the central comparative group
            Central.train(iterations = local_epochs)
            
            ### Train the federated decentralized group
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
                y_pred = server_model.predict(X_test) 
                error = dde.metrics.l2_relative_error(y_test, y_pred)
                l2.append(error)
                # print("Epoch: {}, l2_loss: {}".format(i+1, error))
                
                ## Compute the weight divergence
                WSGD = copy.deepcopy(Central.net.state_dict())
                Wfedavg = copy.deepcopy(server_model.net.state_dict()) 
            
                fc1.append(np.linalg.norm(Wfedavg['linears.0.weight'] - WSGD['linears.0.weight']) / np.linalg.norm(WSGD['linears.0.weight']))
                fc2.append(np.linalg.norm(Wfedavg['linears.1.weight'] - WSGD['linears.1.weight']) / np.linalg.norm(WSGD['linears.1.weight']))
                fc3.append(np.linalg.norm(Wfedavg['linears.2.weight'] - WSGD['linears.2.weight']) / np.linalg.norm(WSGD['linears.2.weight']))
                fc4.append(np.linalg.norm(Wfedavg['linears.3.weight'] - WSGD['linears.3.weight']) / np.linalg.norm(WSGD['linears.3.weight']))

        loss.append(l2)
        wd_fc1.append(fc1)
        wd_fc2.append(fc2)
        wd_fc3.append(fc3)
        wd_fc4.append(fc4)

    np.savez("1D{}.npz".format(iter), loss = loss)
    np.savez("1D{}wd.npz".format(iter), fc1 = wd_fc1, fc2 = wd_fc2, fc3 = wd_fc3, fc4 = wd_fc4)


# # 1D extrapolation baseline

# for i in range(5):
#     loss1 = []
#     loss2 = []
#     for piece in n_pieces:
#         # split data in slice
#         subdata_x, subdata_y = data_assignment_1d(X_train, y_train, piece=piece, n_clients=n_clients)

#         # Slice version implementation
#         client_nets1 = dde.nn.FNN(layer_size,activation,initializer)
#         client_data1 = dde.data.DataSet(X_train=subdata_x[0], y_train=subdata_y[0], X_test=X_test, y_test=y_test)
#         client_model1 = dde.Model(client_data1, client_nets1)
#         client_model1.compile("adam", lr=1e-3, metrics=["l2 relative error"])
#         losshistory1, train_state = client_model1.train(epochs=local_epochs * global_epochs)

#         y_pred1 = client_model1.predict(X_test) 
#         error1 = dde.metrics.l2_relative_error(y_test, y_pred1)
#         loss1.append(error1)
        

#         client_nets2 = dde.nn.FNN(layer_size,activation,initializer)
#         client_data2 = dde.data.DataSet(X_train=subdata_x[1], y_train=subdata_y[1], X_test=X_test, y_test=y_test)
#         client_model2 = dde.Model(client_data2, client_nets2)
#         client_model2.compile("adam", lr=1e-3, metrics=["l2 relative error"])
#         losshistory2, train_state = client_model2.train(epochs=local_epochs * global_epochs)

#         y_pred2 = client_model2.predict(X_test) 
#         error2 = dde.metrics.l2_relative_error(y_test, y_pred2)
#         loss2.append(error2)

#     np.savez("1D_extp_baseline{}.npz".format(i), loss1=loss1, loss2=loss2)
