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
import onedimhelpers as helpers

n_clients = 2
n_pieces = [1,2,3,4,5,8,16]


## parser
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
ap = argparse.ArgumentParser()
ap.add_argument("-le", "--local_epoch", required=True, type=int,
   help="local epoch")
ap.add_argument("-ge", "--global_epoch", required=True, type=int,
   help="global epoch")
ap.add_argument('-run', '--which_run', required=True, type=int, help = 'which repeated runs')
args = vars(ap.parse_args())
local_epochs = args['local_epoch']
global_epochs = args['global_epoch']
which_run = args['which_run']

geom = dde.geometry.Interval(0,np.pi)
num_domain = 32
num_test = 100

def transform(x, y):
    return x + torch.tanh(x) * torch.tanh(np.pi - x) * y
def func(x):
    sol = x + 1 / 8 * np.sin(8 * x)
    for i in range(1, 5):
        sol += 1 / i * np.sin(i * x)
    return sol
layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"


x_train = np.linspace(geom.l, geom.r, num = num_domain).reshape(-1,1)
y_train = func(x_train)

x_test = np.linspace(geom.l, geom.r, num = num_test).reshape(-1,1)
y_test = func(x_test)
mlnotify.start()
baseline_metric = []
federated_metric = []
Extra1_error = []
Extra2_error = []
Weight_divergence1 = []
Weight_divergence2 = []
Weight_divergence3 = []
Weight_divergence4 = []
for piece in n_pieces:
    # split data
    client_x_list, client_y_list = helpers.data_assignment(n_clients, x_train, y_train, piece)
    # server
    server_model = helpers.OneDimPoisson(layer_size,activation,initializer,local_epochs, global_epochs, geom)
    server_model.set_data(x_train)
    server_model.compile()
    # central
    central_model = helpers.OneDimPoisson(layer_size,activation,initializer,local_epochs, global_epochs, geom)
    central_model.set_data(x_train)
    central_model.compile()
    # sync initial weights
    central_model.net.load_state_dict(server_model.net.state_dict())
    ### clients
    CLI1_model = helpers.OneDimPoisson(layer_size,activation,initializer,local_epochs, global_epochs, geom)
    CLI1_model.set_data(client_x_list[0])
    CLI1_model.compile()
    # sync initial weights
    CLI1_model.net.load_state_dict(server_model.net.state_dict())
    CLI2_model = helpers.OneDimPoisson(layer_size,activation,initializer,local_epochs, global_epochs, geom)
    CLI2_model.set_data(client_x_list[1])
    CLI2_model.compile()
    # sync initial weights
    CLI2_model.net.load_state_dict(server_model.net.state_dict())
    ### extrapolation clients
    CLI1_extra_model = helpers.OneDimPoisson(layer_size,activation,initializer,local_epochs, global_epochs, geom)
    CLI1_extra_model.set_data(client_x_list[0])
    CLI1_extra_model.compile()
    # sync initial weights
    CLI1_extra_model.net.load_state_dict(server_model.net.state_dict())
    CLI2_extra_model = helpers.OneDimPoisson(layer_size,activation,initializer,local_epochs, global_epochs, geom)
    CLI2_extra_model.set_data(client_x_list[1])
    CLI2_extra_model.compile()
    # sync initial weights
    CLI2_extra_model.net.load_state_dict(server_model.net.state_dict())
    
    # train extrapolation clients
    CLI1_extra_model.model.train(iterations = global_epochs*local_epochs)
    CLI2_extra_model.model.train(iterations = global_epochs*local_epochs)
    CLI1_extra_error = CLI1_extra_model.get_metric_history()
    CLI2_extra_error = CLI2_extra_model.get_metric_history()
    
    # record
    Extra1_error.append(CLI1_extra_error[-1])
    Extra2_error.append(CLI2_extra_error[-1])

    federated_l2 = []
    # train
    for i in range(global_epochs):
        central_model.train()
        CLI1_model.train()
        CLI2_model.train()

        # aggregate
        
        combined_dict = copy.deepcopy(CLI1_model.model.state_dict())
        for (k1,v1), (k2,v2) in zip(CLI1_model.model.state_dict().items(), CLI2_model.model.state_dict().items()):
            combined_dict[k1] = (v1+v2)/2
        # upload 
        server_model.net.load_state_dict(combined_dict)
        l2_error = server_model.return_l2_error(x_test, y_test)
        federated_l2.append(l2_error)
        # broadcast
        CLI1_model.net.load_state_dict(combined_dict)
        CLI2_model.net.load_state_dict(combined_dict)
        
    # compute WD
    WSGD = copy.deepcopy(central_model.model.state_dict())
    Wfedavg = copy.deepcopy(server_model.model.state_dict())
    WD1 = np.linalg.norm(WSGD['linears.0.weight'].cpu() - Wfedavg['linears.0.weight'].cpu()) / np.linalg.norm(WSGD['linears.0.weight'].cpu())
    WD2 = np.linalg.norm(WSGD['linears.1.weight'].cpu() - Wfedavg['linears.1.weight'].cpu()) / np.linalg.norm(WSGD['linears.1.weight'].cpu())
    WD3 = np.linalg.norm(WSGD['linears.2.weight'].cpu() - Wfedavg['linears.2.weight'].cpu()) / np.linalg.norm(WSGD['linears.2.weight'].cpu())
    WD4 = np.linalg.norm(WSGD['linears.3.weight'].cpu() - Wfedavg['linears.3.weight'].cpu()) / np.linalg.norm(WSGD['linears.3.weight'].cpu())
    Weight_divergence1.append(WD1)
    Weight_divergence2.append(WD2)
    Weight_divergence3.append(WD3)
    Weight_divergence4.append(WD4)
    
        
        

    Central_error = central_model.get_metric_history()

    baseline_metric.append(Central_error[-1])
    federated_metric.append(np.array(federated_l2)[-1])
    
    
    
cur_path = Path().absolute()    
np.savetxt(str(cur_path) + '/NewResults/baseline_error_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), baseline_metric)
np.savetxt(str(cur_path) + '/NewResults/federated_error_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), federated_metric)
np.savetxt(str(cur_path) + '/NewResults/CLI1_error_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), Extra1_error)
np.savetxt(str(cur_path) + '/NewResults/CLI2_error_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), Extra2_error)
np.savetxt(str(cur_path) + '/NewResults/WD1_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), Weight_divergence1)
np.savetxt(str(cur_path) + '/NewResults/WD2_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), Weight_divergence2)
np.savetxt(str(cur_path) + '/NewResults/WD3_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), Weight_divergence3)
np.savetxt(str(cur_path) + '/NewResults/WD4_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), Weight_divergence4)

mlnotify.end()