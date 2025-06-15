# when running the ocde, use DDE_BACKEND=torch python cavity_cavity_fed_u.py 

import os
# os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
import deepxde as dde
import copy
from deepxde.backend import torch


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi,dtype=x.dtype)) * (x + 0.044715 * x ** 3)))

def split_idx(n_clients, piece, idx):
    datalen = len(idx)
    quotient =  datalen // (piece * n_clients)
    residue = datalen % (piece * n_clients)

    idx_list = []

    for j in range(n_clients):
        tempidx = []
        residx = []

        for i in range(piece):
            tempidx.append(idx[(i*n_clients+j)*quotient:(i*n_clients+j+1)*quotient])
    
        extra = 0
        while extra * n_clients + j < residue:
            residx.append(idx[quotient*piece*n_clients+ j + extra * n_clients])
            extra +=1
        idx_list.append(np.hstack((np.array(tempidx).reshape(1, -1), 
                                  np.array(residx).reshape(1,-1))))
        
    return idx_list

n_clients = 2
global_epochs = 80000
local_epochs = 5

n_pieces = [1]
noise_level = [0]

data_train = np.load(f"cavity_train.npz")
data_test = np.load(f"cavity_test.npz")
idx = np.linspace(0,19,20)

X_train = (data_train["branch_train"].astype(np.float32),data_train["trunk_train"].astype(np.float32))
y_train = data_train["u_train"].astype(np.float32)

X_branch_train = X_train[0].reshape(21, 10201, 101)
X_trunk_train = X_train[1].reshape(21, 10201, 2)

X_test = (data_test["branch_test"].astype(np.float32),
          data_test["trunk_test"].astype(np.float32))
y_test = data_test["u_test"].astype(np.float32)


for noise in noise_level:
    y_train += noise * np.random.randn(*y_train.shape).astype(np.float32)

    for iter in range(1):

        loss1 = []
        loss2 = []

        for i in range(len(n_pieces)):

            # federated dataset
            piece = n_pieces[i]
            idx_list = split_idx(n_clients, piece, idx)

            X_train0_client1 = X_branch_train[idx_list[0].squeeze(0).astype(int)].reshape(-1, 101)
            X_train1_client1 = X_trunk_train[idx_list[0].squeeze(0).astype(int)].reshape(-1, 2)
            y_train_client1 = y_train.reshape(21, 10201, 1)[idx_list[0].squeeze(0).astype(int)].reshape(-1, 1)
            X_train0_client2 = X_branch_train[idx_list[1].squeeze(0).astype(int)].reshape(-1, 101)
            X_train1_client2 = X_trunk_train[idx_list[1].squeeze(0).astype(int)].reshape(-1, 2)
            y_train_client2 = y_train.reshape(21, 10201, 1)[idx_list[1].squeeze(0).astype(int)].reshape(-1, 1)
            
            X_train_client1 = (X_train0_client1.astype(np.float32), X_train1_client1.astype(np.float32))
            y_train_client1 = y_train_client1.astype(np.float32)
            X_train_client2 = (X_train0_client2.astype(np.float32), X_train1_client2.astype(np.float32))
            y_train_client2 = y_train_client2.astype(np.float32)
            X_train_clients = (X_train_client1, X_train_client2)
            y_train_clients = (y_train_client1, y_train_client2)


            def loss_func(y_true, y_pred):
                return torch.mean(
                    torch.norm(y_true.view(-1, 101 * 101) - y_pred.view(-1, 101 * 101), dim=1) / torch.norm(
                        y_true.view(-1, 101 * 101), dim=1))


            # define a list of clients as FNNs with same architecture as server
            client_nets = []
            client_models = []
            for i in range(n_clients):
                client_nets.append(dde.nn.deeponet.DeepONet([101, 100, 100, 100],
                                                    [2, 100, 100, 100],
                                                    {"branch": "relu", "trunk": gelu},
                                                    "Glorot normal",
                                                    ))
                client_data = dde.data.Triple(X_train=X_train_clients[i], y_train=y_train_clients[i], X_test=X_test, y_test=y_test)
                model = dde.Model(client_data, client_nets[i])
                client_models.append(model)
                client_models[i].compile("adam", lr=1e-3, loss=loss_func, metrics=["l2 relative error"])
            
            # train
            losshistory1, train_state = client_models[0].train(iterations = local_epochs*global_epochs)
            losshistory2, train_state = client_models[1].train(iterations = local_epochs*global_epochs)
            loss1.append(losshistory1.metrics_test[-1])
            loss2.append(losshistory2.metrics_test[-1])
            client_models[0].save(f"extp_u1_{piece}piece_{noise}noise.ckpt")
            client_models[1].save(f"extp_u2_{piece}piece_{noise}noise.ckpt")
       
