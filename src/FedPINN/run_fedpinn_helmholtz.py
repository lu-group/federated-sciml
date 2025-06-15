import copy
import numpy as np
import deepxde as dde
from deepxde.backend import torch
from data_assignment import data_generation_Hammersly

# Helmholtz equation over 2D square [0, 1]^2
def helmholtz_fed(fname1, fname2, n_clients, n_pieces, local_epochs, global_epochs):

    # Define sine function
    if dde.backend.backend_name == "pytorch":
        sin = dde.backend.pytorch.sin
    elif dde.backend.backend_name == "paddle":
        sin = dde.backend.paddle.sin
    else:
        from deepxde.backend import tf

        sin = tf.sin

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)

        f = k0 ** 2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
        return -dy_xx - dy_yy - k0 ** 2 * y - f


    def func(x):
        return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])


    def transform(x, y):
        res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
        return res * y


    def boundary(_, on_boundary):
        return on_boundary

    # General parameters
    n = 2
    precision_train = 12
    precision_test = 30
    hard_constraint = True

    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    k0 = 2 * np.pi * n
    wave_len = 1 / n

    hx_train = wave_len / precision_train
    nx_train = int(1 / hx_train)

    hx_test = wave_len / precision_test
    nx_test = int(1 / hx_test)

    if hard_constraint == True:
        bc = []
    else:
        bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

    layer_size = [2] + [64] * 3 + [1]
    activation = "sin"
    initializer = "Glorot uniform"

    
    loss = []
    wd_fc1 = []
    wd_fc2 = []
    wd_fc3 = []
    wd_fc4 = []

    for piece in n_pieces:
        subdata_x = data_generation_Hammersly(piece, nx_train, ((0, 0), (1, 1)))
        # Server PINN
        server_net = dde.nn.FNN(layer_size, activation, initializer)
        server_net.apply_output_transform(transform)
        server_data = dde.data.PDE(geom, pde, bc, num_domain=nx_train ** 2, num_boundary=4 * nx_train,
                                    solution=func, num_test=nx_test ** 2)
        server_model = dde.Model(server_data, server_net)
        server_model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
        
        # central PINN
        central_net = dde.nn.FNN(layer_size, activation, initializer)
        central_net.apply_output_transform(transform)
        central_data = dde.data.PDE(geom, pde, bc, num_domain=nx_train ** 2, num_boundary=4 * nx_train,
                                    solution=func, num_test=nx_test ** 2)
        Central = dde.Model(central_data, central_net)
        Central.compile("adam", lr=1e-3, metrics=["l2 relative error"])

        # Client PINN
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.FNN(layer_size,activation,initializer))
            client_nets[i].apply_output_transform(transform)
            client_data = dde.data.PDE(geom, pde,bc, num_domain=0, num_boundary=4 * nx_train,
                                        solution=func, num_test=nx_test ** 2, anchors=subdata_x[i].reshape(-1,2))
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
                losshistory, train_state = server_model.train(iterations=0)
                l2.append(losshistory.metrics_test[-1])
                
                ## Compute the weight divergence
                WSGD = copy.deepcopy(Central.net.state_dict())
                Wfedavg = copy.deepcopy(server_model.net.state_dict()) 
            
                fc1.append(np.linalg.norm(Wfedavg['linears.0.weight'].cpu().numpy() - WSGD['linears.0.weight'].cpu().numpy()) / np.linalg.norm(WSGD['linears.0.weight'].cpu().numpy()))
                fc2.append(np.linalg.norm(Wfedavg['linears.1.weight'].cpu().numpy() - WSGD['linears.1.weight'].cpu().numpy()) / np.linalg.norm(WSGD['linears.1.weight'].cpu().numpy()))
                fc3.append(np.linalg.norm(Wfedavg['linears.2.weight'].cpu().numpy() - WSGD['linears.2.weight'].cpu().numpy()) / np.linalg.norm(WSGD['linears.2.weight'].cpu().numpy()))
                fc4.append(np.linalg.norm(Wfedavg['linears.3.weight'].cpu().numpy() - WSGD['linears.3.weight'].cpu().numpy()) / np.linalg.norm(WSGD['linears.3.weight'].cpu().numpy()))

        loss.append(l2)
        wd_fc1.append(fc1)
        wd_fc2.append(fc2)
        wd_fc3.append(fc3)
        wd_fc4.append(fc4)
        # np.savez("1D{}_wd_piece{}_tanh.npz".format(iter,piece), fc1 = fc1, fc2 = fc2, fc3 = fc3, fc4 = fc4)
    np.savez(fname1, loss = loss)
    np.savez(fname2, fc1 = wd_fc1, fc2 = wd_fc2, fc3 = wd_fc3, fc4 = wd_fc4)

def helmholtz_extp(fname, n_clients, n_pieces, local_epochs, global_epochs):

    # Define sine function
    if dde.backend.backend_name == "pytorch":
        sin = dde.backend.pytorch.sin
    elif dde.backend.backend_name == "paddle":
        sin = dde.backend.paddle.sin
    else:
        from deepxde.backend import tf

        sin = tf.sin

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)

        f = k0 ** 2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
        return -dy_xx - dy_yy - k0 ** 2 * y - f


    def func(x):
        return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])


    def transform(x, y):
        res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
        return res * y


    def boundary(_, on_boundary):
        return on_boundary

    # General parameters
    n = 2
    precision_train = 12
    precision_test = 30
    hard_constraint = True

    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    k0 = 2 * np.pi * n
    wave_len = 1 / n

    hx_train = wave_len / precision_train
    nx_train = int(1 / hx_train)

    hx_test = wave_len / precision_test
    nx_test = int(1 / hx_test)

    if hard_constraint == True:
        bc = []
    else:
        bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

    layer_size = [2] + [64] * 3 + [1]
    activation = "sin"
    initializer = "Glorot uniform"

    loss1 = []
    loss2 = []

    uniform_data = np.linspace(0, 1, nx_test)
    X, Y = np.meshgrid(uniform_data, uniform_data)
    x_test = np.vstack([X.ravel(), Y.ravel()]).T

    for piece in n_pieces:
        subdata_x = data_generation_Hammersly(piece, nx_train, ((0, 0), (1, 1)))

        # Client PINN
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.FNN(layer_size,activation,initializer))
            client_nets[i].apply_output_transform(transform)
            client_data = dde.data.PDE(geom, pde,bc, num_domain=0, num_boundary=4 * nx_train,
                                        solution=func, num_test=nx_test ** 2, anchors=subdata_x[i].reshape(-1,2))
            model = dde.Model(client_data, client_nets[i])
            client_models.append(model)
            client_models[i].compile("adam", lr=1e-3, metrics=["l2 relative error"])

        client_models[0].train(iterations = local_epochs * global_epochs)
        y_pred1 =client_models[0].predict(x_test)
        loss1.append(dde.metrics.l2_relative_error(func(x_test), y_pred1))
        client_models[1].train(iterations = local_epochs * global_epochs)
        y_pred2 =client_models[1].predict(x_test)
        loss2.append(dde.metrics.l2_relative_error(func(x_test), y_pred2))
       
    np.savez(fname, loss1 = loss1, loss2 = loss2)


n_clients = 2
local_epochs = 5
global_epochs = 2000

# Helmholtz
n_pieces = [1,2,4,5,7,11,23]

for i in range(5):
    # helmholtz_fed("helmholtz.npz".format(i), "helmholtz_wd.npz".format(i), n_clients, n_pieces, local_epochs, global_epochs)
    helmholtz_extp(f"helmholtz_extp.npz", n_clients, n_pieces, local_epochs, global_epochs)
