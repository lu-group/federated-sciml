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

# true values
C1true = 1.0
C2true = 0.01

n_clients = 2
n_pieces = [1,2,3,4,5,10]
is_split_x = True


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


# Rectangular
Lx_min, Lx_max = 1.0, 8.0
Ly_min, Ly_max = -2.0, 2.0
# Spatial domain: X × Y = [1, 8] × [−2, 2]
space_domain = dde.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])
# Time domain: T = [0, 1]
time_domain = dde.geometry.TimeDomain(0, 1)
# Spatio-temporal domain
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)

# Get the training data: num = 7000
[ob_x, ob_y, ob_t, ob_u, ob_v, ob_p] = helpers.load_training_data(num=7000,x1=Lx_min, x2=Lx_max, y1=Ly_min, y2=Ly_max)
ob_xyt = np.hstack((ob_x, ob_y, ob_t))
observe_u = dde.icbc.PointSetBC(ob_xyt, ob_u, component=0)
observe_v = dde.icbc.PointSetBC(ob_xyt, ob_v, component=1)

# Neural Network setup
layer_size = [3] + [50] * 6 + [3]
activation = "tanh"
initializer = "Glorot uniform"
mlnotify.start()
baseline = []
federated = []
for piece in n_pieces:
    # split data
    client_data_list, observe_u_list, observe_v_list = helpers.client_data_generator(geom=space_domain, n_clients=n_clients, piece=piece, is_split_x=is_split_x)
    # directly define server as a FNN
    server_model = helpers.InverseNS(layer_size,activation,initializer,local_epochs, global_epochs, geomtime)
    server_model.set_data([observe_u, observe_v])
    server_model.set_variable('server_piece={}_run={}.dat'.format(str(piece),str(which_run)))
    server_model.compile()
    central_model = helpers.InverseNS(layer_size,activation,initializer,local_epochs, global_epochs, geomtime)
    central_model.set_data([observe_u, observe_v])
    central_model.set_variable('central_piece={}_run={}.dat'.format(str(piece),str(which_run)))
    central_model.compile()
    ### clients
    CLI1_model = helpers.InverseNS(layer_size,activation,initializer,local_epochs, global_epochs, geomtime)
    CLI1_model.set_data([observe_u_list[0], observe_v_list[0]])
    CLI1_model.set_variable('CLI1_piece={}_run={}.dat'.format(str(piece),str(which_run)))
    CLI1_model.compile()
    CLI2_model = helpers.InverseNS(layer_size,activation,initializer,local_epochs, global_epochs, geomtime)
    CLI2_model.set_data([observe_u_list[1], observe_v_list[1]])
    CLI2_model.set_variable('CLI2_piece={}_run={}.dat'.format(str(piece),str(which_run)))
    CLI2_model.compile()


    # train
    for i in range(global_epochs):
        central_model.train()
        CLI1_model.train()
        CLI2_model.train()

        # aggregate
        server_model.C1.data = (CLI1_model.C1.data + CLI2_model.C1.data) /2
        server_model.C2.data = (CLI1_model.C2.data + CLI2_model.C2.data) /2
        
        combined_dict = copy.deepcopy(CLI1_model.model.state_dict())
        for (k1,v1), (k2,v2) in zip(CLI1_model.model.state_dict().items(), CLI2_model.model.state_dict().items()):
            combined_dict[k1] = (v1+v2)/2

        # broadcast
        
        CLI1_model.net.load_state_dict(combined_dict)
        CLI2_model.net.load_state_dict(combined_dict)
        CLI1_model.C1.data = server_model.C1.data
        CLI1_model.C2.data = server_model.C2.data
        CLI2_model.C1.data = server_model.C1.data
        CLI2_model.C2.data = server_model.C2.data
    CLI1_loss = CLI1_model.get_loss_history()
    CLI2_loss = CLI2_model.get_loss_history()
    federated_loss = (CLI1_loss + CLI2_loss) / 2
    Central_loss = central_model.get_loss_history()
    baseline.append(np.min(Central_loss))
    federated.append(np.min(federated_loss))
    
    
cur_path = Path().absolute()    
np.savetxt(str(cur_path) + '/baseline_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs),str(which_run)), baseline)
np.savetxt(str(cur_path) + '/federated_ge={}_le={}_run={}.txt'.format(str(global_epochs), str(local_epochs),str(which_run)), federated)
mlnotify.end()