import copy
import numpy as np
import deepxde as dde
from deepxde.backend import torch
from scipy.io import loadmat
from deepxde.backend import torch
from data_assignment import quasirandom, data_assignment_2d_slice


# Allen Cahn equation, x in [-1, 1], t in [0, 1]
def allen_cahn_fed(fname1, fname2, n_clients, n_pieces, local_epochs, global_epochs):

    def gen_testdata():
        data = loadmat("usol_D_0.001_k_5.mat")

        t = data["t"]

        x = data["x"]

        u = data["u"]

        dt = dx = 0.01

        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = u.flatten()[:, None]
        return X, y

    def pde(x, y):
        u = y
        du_xx = dde.grad.hessian(y, x, i=0, j=0)
        du_t = dde.grad.jacobian(y, x, i=0, j=1)
        return du_t - 0.001 * du_xx + 5 * (u ** 3 - u)

    def transform(x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]
        return t_in * (1 + x_in) * (1 - x_in) * y + torch.square(x_in) * torch.cos(np.pi * x_in)
   
    # General parameters
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    X, y_true = gen_testdata()

    layer_size = [2] + [64] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"

    loss = []
    wd_fc1 = []
    wd_fc2 = []
    wd_fc3 = []
    # wd_fc4 = []

    for piece in n_pieces:
        subdata_x = data_assignment_2d_slice([-1,1], [0,1], num=80, n_cut=piece, n_clients=n_clients)
        sample_pts = np.concatenate((subdata_x[0], subdata_x[1]))
        # Server PINN
        server_net = dde.nn.FNN(layer_size, activation, initializer)
        server_net.apply_output_transform(transform)
        server_data = dde.data.TimePDE(geomtime, pde, [], num_domain=0, num_boundary=400, num_initial=800, 
                                       train_distribution='uniform', anchors=sample_pts)
        server_model = dde.Model(server_data, server_net)
        server_model.compile("adam", lr=1e-3)
        
        # central PINN
        central_net = dde.nn.FNN(layer_size, activation, initializer)
        central_net.apply_output_transform(transform)
        central_data = dde.data.TimePDE(geomtime, pde, [], num_domain=0, num_boundary=400, num_initial=800,
                                        train_distribution='uniform', anchors=sample_pts)
        Central = dde.Model(central_data, central_net)
        Central.compile("adam", lr=1e-3)

        # Client PINN
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.FNN(layer_size,activation,initializer))
            client_nets[i].apply_output_transform(transform)
            client_data = dde.data.TimePDE(geomtime, pde, [], num_domain=0, num_boundary = 400, num_initial=800, 
                                           train_distribution='uniform', anchors=subdata_x[i])
            model = dde.Model(client_data, client_nets[i])
            client_models.append(model)
            client_models[i].compile("adam", lr=1e-3)
            
        # train
        l2 = []
        fc1 = []
        fc2 = []
        fc3 = []
        # fc4 = []
        for i in range(global_epochs):
            # Central training
            Central.train(iterations = local_epochs)
            
            # Federated training
            for j in range(len(client_models)):
                train_state = client_models[j].train(iterations=local_epochs) 

            # aggregate
            combined_dict = copy.deepcopy(client_models[0].state_dict())
            for (k1,v1), (k2,v2) in zip(client_models[0].state_dict().items(), client_models[1].state_dict().items()):
                combined_dict[k1] = (v1+v2)/2

            # broadcast
            for j in range(len(client_models)):
                client_nets[j].load_state_dict(combined_dict)
            
            # error and weight divergence
            if i==0 or (i+1)%1000 == 0:
                server_net.load_state_dict(combined_dict)
                y_pred = server_model.predict(X)
                print("L2 relative error is: ", dde.metrics.l2_relative_error(y_true, y_pred))
                l2.append(dde.metrics.l2_relative_error(y_true, y_pred))
                
                ## Compute the weight divergence
                WSGD = copy.deepcopy(Central.net.state_dict())
                Wfedavg = copy.deepcopy(server_model.net.state_dict()) 
            
                fc1.append(np.linalg.norm(Wfedavg['linears.0.weight'].cpu().numpy() - WSGD['linears.0.weight'].cpu().numpy()) / np.linalg.norm(WSGD['linears.0.weight'].cpu().numpy()))
                fc2.append(np.linalg.norm(Wfedavg['linears.1.weight'].cpu().numpy() - WSGD['linears.1.weight'].cpu().numpy()) / np.linalg.norm(WSGD['linears.1.weight'].cpu().numpy()))
                fc3.append(np.linalg.norm(Wfedavg['linears.2.weight'].cpu().numpy() - WSGD['linears.2.weight'].cpu().numpy()) / np.linalg.norm(WSGD['linears.2.weight'].cpu().numpy()))

        loss.append(l2)
        wd_fc1.append(fc1)
        wd_fc2.append(fc2)
        wd_fc3.append(fc3)

    np.savez(fname1, loss = loss)
    np.savez(fname2, fc1 = wd_fc1, fc2 = wd_fc2, fc3 = wd_fc3)

def allen_cahn_extp(fname, n_clients, n_pieces, local_epochs, global_epochs):

    def gen_testdata():
        data = loadmat("usol_D_0.001_k_5.mat")

        t = data["t"]

        x = data["x"]

        u = data["u"]

        dt = dx = 0.01

        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = u.flatten()[:, None]
        return X, y

    def pde(x, y):
        u = y
        du_xx = dde.grad.hessian(y, x, i=0, j=0)
        du_t = dde.grad.jacobian(y, x, i=0, j=1)
        return du_t - 0.001 * du_xx + 5 * (u ** 3 - u)

    def transform(x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]
        return t_in * (1 + x_in) * (1 - x_in) * y + torch.square(x_in) * torch.cos(np.pi * x_in)
   
    # General parameters
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    X, y_true = gen_testdata()

    layer_size = [2] + [64] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"

    loss1 = []
    loss2 = []
  
    for piece in n_pieces:
        subdata_x = data_assignment_2d_slice([-1,1], [0,1], num=80, n_cut=piece, n_clients=2)

        # Client PINN
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.FNN(layer_size,activation,initializer))
            client_nets[i].apply_output_transform(transform)
            client_data = dde.data.TimePDE(geomtime, pde, [], num_domain=0, num_boundary = 400, num_initial=800, 
                                           train_distribution='uniform', anchors=subdata_x[i])
            model = dde.Model(client_data, client_nets[i])
            client_models.append(model)
            client_models[i].compile("adam", lr=1e-3)
        
        client_models[0].train(iterations = 30000)
        y_pred1 =client_models[0].predict(X)
        loss1.append(dde.metrics.l2_relative_error(y_true, y_pred1))
        client_models[1].train(iterations = 30000)
        y_pred2 =client_models[1].predict(X)
        loss2.append(dde.metrics.l2_relative_error(y_true, y_pred2))
    np.savez(fname, loss1 = loss1, loss2 = loss2)

n_clients = 2
local_epochs = 5
global_epochs = 10000

# Allen-Cahn
n_pieces = [1, 2, 3, 4, 5, 10, 16, 20]

for i in range(5):
    allen_cahn_fed("allencahn{}.npz".format(i),"allencahn{}_wd.npz".format(i),  n_clients, n_pieces, local_epochs, global_epochs)
    allen_cahn_extp("allencahn{}_extp.npz".format(i),  n_clients, n_pieces, local_epochs, global_epochs)
