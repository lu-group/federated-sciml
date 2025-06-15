import copy
import numpy as np
import deepxde as dde
from deepxde.backend import torch
from data_assignment import data_assignmentX_1d

# Euler beam
def eulerbeam_fed(fname1, fname2, n_clients, n_pieces, local_epochs, global_epochs):

    def ddy(x, y):
        return dde.grad.hessian(y, x)

    def dddy(x, y):
        return dde.grad.jacobian(ddy(x, y), x)

    def pde(x, y):
        dy_xx = ddy(x, y)
        dy_xxxx = dde.grad.hessian(dy_xx, x)
        return dy_xxxx + 1


    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)


    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)


    def func(x):
        return -(x ** 4) / 24 + x ** 3 / 6 - x ** 2 / 4

    def transform(x,y ):
        # res = x * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
        # return res * y
        return (x ** 2) * y


    sample_pts = np.loadtxt("Euler_beam.txt")[:,None]
    geom = dde.geometry.Interval(0, 1)

    bc1 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l)
    bc2 = dde.icbc.NeumannBC(geom, lambda x: 0, boundary_l)
    bc3 = dde.icbc.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)
    bc4 = dde.icbc.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)

    layer_size = [1] + [20] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"

    loss = []
    wd_fc1 = []
    wd_fc2 = []
    wd_fc3 = []
    wd_fc4 = []

    for piece in n_pieces:
        subdata_x = data_assignmentX_1d(sample_pts, piece=piece, n_clients=n_clients)

        # Server PINN
        server_net = dde.nn.FNN(layer_size, activation, initializer)
        server_data = dde.data.PDE(geom, pde,[bc3, bc4], num_domain=0, num_boundary=2, 
                            solution=func, num_test=100, anchors=sample_pts)
        server_model = dde.Model(server_data, server_net)
        server_net.apply_output_transform(transform)
        server_model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
        
        # central PINN
        central_net = dde.nn.FNN(layer_size, activation, initializer)
        central_data = dde.data.PDE(geom, pde,[ bc3, bc4], num_domain=0, num_boundary=2, 
                            solution=func, num_test=100, anchors=sample_pts)
        Central = dde.Model(central_data, central_net)
        central_net.apply_output_transform(transform)
        Central.compile("adam", lr=1e-3, metrics=["l2 relative error"])

        # Client PINN
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.FNN(layer_size,activation,initializer))
            client_data = dde.data.PDE(geom, pde,[bc3, bc4], num_domain=0, num_boundary=2, 
                                       solution=func, num_test=100, anchors=subdata_x[i])
            model = dde.Model(client_data, client_nets[i])
            client_models.append(model)
            client_models[i].compile("adam", lr=1e-3, metrics=["l2 relative error"])
            client_nets[i].apply_output_transform(transform)

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


def eulerbeam_extp(fname, n_clients, n_pieces, local_epochs, global_epochs):

    def ddy(x, y):
        return dde.grad.hessian(y, x)

    def dddy(x, y):
        return dde.grad.jacobian(ddy(x, y), x)

    def pde(x, y):
        dy_xx = ddy(x, y)
        dy_xxxx = dde.grad.hessian(dy_xx, x)
        return dy_xxxx + 1


    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)


    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)


    def func(x):
        return -(x ** 4) / 24 + x ** 3 / 6 - x ** 2 / 4

    def transform(x,y):
        # res = x * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
        # return res * y
        return (x ** 2) * y


    sample_pts = np.loadtxt("Euler_beam.txt")[:,None]
    geom = dde.geometry.Interval(0, 1)

    bc1 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l)
    bc2 = dde.icbc.NeumannBC(geom, lambda x: 0, boundary_l)
    bc3 = dde.icbc.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)
    bc4 = dde.icbc.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)

    layer_size = [1] + [20] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"

    loss1 = []
    loss2 = []

    for piece in n_pieces:
        subdata_x = data_assignmentX_1d(sample_pts, piece=piece, n_clients=n_clients)

        # Client PINN
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.FNN(layer_size,activation,initializer))
            client_data = dde.data.PDE(geom, pde,[bc3, bc4], num_domain=0, num_boundary=2, 
                                       solution=func, num_test=100, anchors=subdata_x[i])
            model = dde.Model(client_data, client_nets[i])
            client_models.append(model)
            client_models[i].compile("adam", lr=1e-3, metrics=["l2 relative error"])
            client_nets[i].apply_output_transform(transform)
        
        losshistory1, train_state = client_models[0].train(iterations = local_epochs*global_epochs)
        losshistory2, train_state = client_models[1].train(iterations = local_epochs*global_epochs)
        loss1.append(losshistory1.metrics_test[-1] )
        loss2.append(losshistory2.metrics_test[-1] )

    np.savez(fname, loss1 = loss1, loss2 = loss2)

n_clients = 2
local_epochs = 5
global_epochs = 1000

# Eulerbeam
n_pieces = [1,2,3,4,5,6,7,8,9,10,25]

for i in range(5):
    eulerbeam_fed("finalresults/eulerbeam{}.npz".format(i),"finalresults/eulerbeam{}_wd.npz".format(i), n_clients, n_pieces, local_epochs, global_epochs)
    eulerbeam_extp("finalresults/eulerbeam{}_extp.npz".format(i), n_clients, n_pieces, local_epochs, global_epochs)
