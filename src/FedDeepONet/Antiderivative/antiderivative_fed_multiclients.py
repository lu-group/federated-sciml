import numpy as np
import deepxde as dde
from deepxde.backend import torch
from scipy.io import loadmat
import skopt
from pathos.pools import ProcessPool
from scipy import interpolate
from scipy.integrate import solve_ivp
from sklearn import gaussian_process as gp
import copy
from deepxde.data.function_spaces import PowerSeries, Chebyshev, GRF, GRF_KL

def generator_ode(ode, space, T, m, n, num, dim_x):
    """Generate `n` random feature vectors.
    """
    # n: number of test or train
    # num: number of point chosen

    print("Generating operator data...", flush=True)
    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values = generator_sensorvals(space, n, sensors)
    s = list(map(ode, sensor_values))
    x = np.linspace(0, T, num=num)[:, None]
    # X = [sensor_values, x]
    y = np.array(s)
    return sensor_values, x, y


def int_index(x, t):
    mat = np.linspace(0, 1, x)
    return int(t / mat[1])


def ode(sensor_values):
    def model(t, s):
        u = lambda t: sensor_values[t]
        return u(int_index(100, t))

    res = solve_ivp(model, [0, 1], s0, method="RK45", t_eval=np.linspace(0, 1, 100))
    return res.y[0, :]



def generator_sensorvals(space, num, sensors):
    '''
    space (string): type of functional space
    num (int): number of functions
    Other hyperparameters are the default setting in deepxde
    '''
    if space == 'PowerSeries':
        space = PowerSeries(N=100, M=1)
        features = space.random(num)
        return space.eval_batch(features, sensors)
    
    elif space == 'Chebyshev':
        space = Chebyshev(N=100, M=1)
        features = space.random(num)
        return space.eval_batch(features, sensors)
    
    elif space == 'GRF':
        space = GRF(1, length_scale=0.1, N=1000, interp="cubic")
        features = space.random(num)
        return space.eval_batch(features, sensors)
    
    elif space == 'GRF_KL':
        space = GRF_KL(1, length_scale=0.1, N=100, interp="cubic")
        features = space.random(num)
        return space.eval_batch(features, sensors)
    
    else:
        print( "FUNCTION SPACE NOT FOUND.")

m = 100
dim_x = 1
T = 1
global_epochs = 2000
local_epochs = 5
num_train = 500
num_test = 1000

num = 100
s0 = [0]
n_clients = 4
spaces  = ['PowerSeries', 'Chebyshev', 'GRF', 'GRF_KL']
loss = []

for iter in range(5):

    space1 = spaces[0]
    space2 = spaces[1]
    space3 = spaces[2]
    space4 = spaces[3]
    
    # federated dataset
    X_train0_client1, X_train1_client1, y_train_client1 = generator_ode(ode, space1, T, m, num_train//n_clients, num, dim_x)
    X_train0_client2, X_train1_client2, y_train_client2 = generator_ode(ode, space2, T, m, num_train//n_clients, num, dim_x)
    X_train0_client3, X_train1_client3, y_train_client3 = generator_ode(ode, space3, T, m, num_train//n_clients, num, dim_x)
    X_train0_client4, X_train1_client4, y_train_client4 = generator_ode(ode, space4, T, m, num_train//n_clients, num, dim_x)
    
    X_train_client1 = (X_train0_client1.astype(np.float32), X_train1_client1.astype(np.float32))
    y_train_client1 = y_train_client1.astype(np.float32)
    X_train_client2 = (X_train0_client2.astype(np.float32), X_train1_client2.astype(np.float32))
    y_train_client2 = y_train_client2.astype(np.float32)
    X_train_client3 = (X_train0_client3.astype(np.float32), X_train1_client3.astype(np.float32))
    y_train_client3 = y_train_client3.astype(np.float32)
    X_train_client4 = (X_train0_client4.astype(np.float32), X_train1_client4.astype(np.float32))
    y_train_client4 = y_train_client4.astype(np.float32)
    X_train_clients = (X_train_client1, X_train_client2, X_train_client3, X_train_client4)
    y_train_clients = (y_train_client1, y_train_client2, y_train_client3, y_train_client4)

    # central dataset
    X_train0_central = np.vstack((X_train0_client1, X_train0_client2, X_train0_client3, X_train0_client4))
    X_train1_central = X_train1_client1 # location points are the same
    y_train_central = np.vstack((y_train_client1, y_train_client2, y_train_client3, y_train_client4))
    X_train = (X_train0_central.astype(np.float32), X_train1_central.astype(np.float32))
    y_train = y_train_central.astype(np.float32)

    # test dataset
    X_test0_client1, X_test1_client1, y_test_client1 = generator_ode(ode, space1, T, m, num_test//n_clients, num, dim_x)
    X_test0_client2, X_test1_client2, y_test_client2 = generator_ode(ode, space2, T, m, num_test//n_clients, num, dim_x)
    X_test0_client3, X_test1_client3, y_test_client3 = generator_ode(ode, space3, T, m, num_test//n_clients, num, dim_x)
    X_test0_client4, X_test1_client4, y_test_client4 = generator_ode(ode, space4, T, m, num_test//n_clients, num, dim_x)
    X_test0 = np.vstack((X_test0_client1, X_test0_client2, X_test0_client3, X_test0_client4))
    X_test1 = X_test1_client1
    y_test = np.vstack((y_test_client1, y_test_client2, y_test_client3, y_test_client4))
    X_test= (X_test0.astype(np.float32), X_test1.astype(np.float32))
    y_test = y_test.astype(np.float32)

    # Server model
    server_net = dde.nn.DeepONetCartesianProd([m, 40, 40], [dim_x, 40, 40], "relu", "Glorot normal")
    data = dde.data.TripleCartesianProd(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    server_model = dde.Model(data, server_net)
    server_model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    
    # Define a centralized training comparison group
    central_net = dde.nn.DeepONetCartesianProd([m, 40, 40], [dim_x, 40, 40], "relu", "Glorot normal")
    central_data = dde.data.TripleCartesianProd(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    Central = dde.Model(central_data, central_net)
    Central.compile("adam", lr=1e-3, metrics=["l2 relative error"])

    # define a list of clients as FNNs with same architecture as server
    client_nets = []
    client_models = []
    for i in range(n_clients):
        client_nets.append(dde.nn.DeepONetCartesianProd([m, 40, 40], [dim_x, 40, 40], "relu", "Glorot normal"))
        client_data = dde.data.TripleCartesianProd(X_train=X_train_clients[i], y_train=y_train_clients[i], X_test=X_test, y_test=y_test)
        model = dde.Model(client_data, client_nets[i])
        client_models.append(model)
        client_models[i].compile("adam", lr=1e-3, metrics=["l2 relative error"])
    print(np.shape(client_models))
    # train
    l2 = []

    for i in range(global_epochs):
        # Train the central model
        Central.train(iterations = local_epochs)
        
        # Train the federated models
        for j in range(len(client_models)):
            train_state = client_models[j].train(iterations=local_epochs) 

        # aggregate
        combined_dict = copy.deepcopy(client_models[0].state_dict())
        for (k1,v1), (k2,v2), (k3,v3), (k4,v4) in zip(client_models[0].state_dict().items(), 
                                             client_models[1].state_dict().items(),
                                             client_models[2].state_dict().items(),
                                             client_models[3].state_dict().items()):
            combined_dict[k1] = (v1+v2+v3+v4)/4

        # broadcast
        for j in range(len(client_models)):
            client_nets[j].load_state_dict(combined_dict)
        
        # (optimal) test and calculate error
        if i==0 or (i+1)%1000 == 0:
            server_net.load_state_dict(combined_dict)
            y_pred = server_model.predict(X_test) 
            error = dde.metrics.l2_relative_error(y_test, y_pred)
            l2.append(error)
    loss.append(l2)

np.savez("antiderivative52k_fed_{}clients.npz".format(n_clients), loss = loss)
