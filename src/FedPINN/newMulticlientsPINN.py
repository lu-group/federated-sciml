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
import helpers


#### banchmark : local ==5 , global == 10000, nc = 3

## parser
ap = argparse.ArgumentParser()
ap.add_argument("-le", "--local_epoch", required=True, type=int,
   help="local epoch")
ap.add_argument("-ge", "--global_epoch", required=True, type=int,
   help="global epoch")
ap.add_argument('-nc', '--number_of_clients', required=True, type=int, help = 'how many clients')
ap.add_argument('-run', '--which_run', required=True, type=int, help = 'which repeated runs')
args = vars(ap.parse_args())
local_epochs = args['local_epoch']
global_epochs = args['global_epoch']
n_clients = args['number_of_clients']
which_run = args['which_run']

n_pieces = [1,2,3,4,5,6,10,20] ## for 3 clients
# n_pieces = [1,2,3,4,5,10] ## for 5 clients



geom = dde.geometry.Interval(-1.0, 1.0)
timedomain = dde.geometry.TimeDomain(0.0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
num_domain = 8000
d = 0.001
is_split_geometry = True
split_method = 'Spatial' if is_split_geometry else 'Temporal'


layer_size = [2] + [64] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"

function_name = 'Allen_Cahn'

x_test, y_test = helpers.gen_testdata()


mlnotify.start()
baseline_metric = []
federated_metric = []
Extra1_error = []
Extra2_error = []
Extra3_error = []
for piece in n_pieces:
    # split data
    client_x_list = helpers.data_generation_Hammersly(piece, num_domain, n_clients, geomtime)
    server_data = np.concatenate(client_x_list,axis = 0)
    # server
    server_model = helpers.Multi_Allen_Cahn(layer_size,activation,initializer,local_epochs, global_epochs, geomtime)
    server_model.set_data(server_data)
    server_model.compile()
    # central
    central_model = helpers.Multi_Allen_Cahn(layer_size,activation,initializer,local_epochs, global_epochs, geomtime)
    central_model.set_data(server_data)
    central_model.compile()
    # sync initial weights
    central_model.net.load_state_dict(server_model.net.state_dict())
    ### clients
    
    ## CLI1
    CLI1_model = helpers.Multi_Allen_Cahn(layer_size,activation,initializer,local_epochs, global_epochs, geomtime)
    CLI1_model.set_data(client_x_list[0])
    CLI1_model.compile()
    # sync initial weights
    CLI1_model.net.load_state_dict(server_model.net.state_dict())
    
    ## CLI2
    CLI2_model = helpers.Multi_Allen_Cahn(layer_size,activation,initializer,local_epochs, global_epochs, geomtime)
    CLI2_model.set_data(client_x_list[1])
    CLI2_model.compile()
    # sync initial weights
    CLI2_model.net.load_state_dict(server_model.net.state_dict())
    
    ## CLI3
    CLI3_model = helpers.Multi_Allen_Cahn(layer_size,activation,initializer,local_epochs, global_epochs, geomtime)
    CLI3_model.set_data(client_x_list[2])
    CLI3_model.compile()
    # sync initial weights
    CLI3_model.net.load_state_dict(server_model.net.state_dict())
    
    ### extrapolation clients
    CLI1_extra_model = helpers.Multi_Allen_Cahn(layer_size,activation,initializer,local_epochs, global_epochs, geomtime)
    CLI1_extra_model.set_data(client_x_list[0])
    CLI1_extra_model.compile()
    # sync initial weights
    CLI1_extra_model.net.load_state_dict(server_model.net.state_dict())
    
    CLI2_extra_model = helpers.Multi_Allen_Cahn(layer_size,activation,initializer,local_epochs, global_epochs, geomtime)
    CLI2_extra_model.set_data(client_x_list[1])
    CLI2_extra_model.compile()
    # sync initial weights
    CLI2_extra_model.net.load_state_dict(server_model.net.state_dict())
    
    CLI3_extra_model = helpers.Multi_Allen_Cahn(layer_size,activation,initializer,local_epochs, global_epochs, geomtime)
    CLI3_extra_model.set_data(client_x_list[2])
    CLI3_extra_model.compile()
    # sync initial weights
    CLI3_extra_model.net.load_state_dict(server_model.net.state_dict())    
    # train extrapolation clients
    CLI1_extra_model.model.train(iterations = global_epochs*local_epochs)
    CLI2_extra_model.model.train(iterations = global_epochs*local_epochs)
    CLI3_extra_model.model.train(iterations = global_epochs*local_epochs)
    CLI1_extra_error = CLI1_extra_model.return_l2_error(x_test, y_test)
    CLI2_extra_error = CLI2_extra_model.return_l2_error(x_test, y_test)
    CLI3_extra_error = CLI3_extra_model.return_l2_error(x_test, y_test)
    
    # record
    Extra1_error.append(CLI1_extra_error)
    Extra2_error.append(CLI2_extra_error)
    Extra3_error.append(CLI3_extra_error)

    federated_l2 = []
    # train
    for i in range(global_epochs):
        central_model.train()
        CLI1_model.train()
        CLI2_model.train()
        CLI3_model.train()

        # aggregate
        
        combined_dict = copy.deepcopy(CLI1_model.model.state_dict())
        for (k1,v1), (k2,v2),(k3,v3) in zip(CLI1_model.model.state_dict().items(), CLI2_model.model.state_dict().items(), CLI3_model.model.state_dict().items()):
            combined_dict[k1] = (v1+v2+v3)/3
        # upload 
        server_model.net.load_state_dict(combined_dict)
        l2_error = server_model.return_l2_error(x_test, y_test)
        federated_l2.append(l2_error)
        # broadcast
        CLI1_model.net.load_state_dict(combined_dict)
        CLI2_model.net.load_state_dict(combined_dict)
        CLI3_model.net.load_state_dict(combined_dict)

        
        

    Central_error = central_model.return_l2_error(x_test, y_test)

    baseline_metric.append(Central_error)
    federated_metric.append(np.array(federated_l2)[-1])
    
    
    
cur_path = Path().absolute()    
np.savetxt(str(cur_path) + '/NewResults/baseline_error_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), baseline_metric)
np.savetxt(str(cur_path) + '/NewResults/federated_error_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), federated_metric)
np.savetxt(str(cur_path) + '/NewResults/CLI1_error_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), Extra1_error)
np.savetxt(str(cur_path) + '/NewResults/CLI2_error_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), Extra2_error)
np.savetxt(str(cur_path) + '/NewResults/CLI3_error_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs), str(which_run)), Extra3_error)

mlnotify.end()