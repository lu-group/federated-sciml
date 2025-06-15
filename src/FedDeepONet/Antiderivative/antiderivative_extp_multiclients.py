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
n_clients = 2
global_epochs = 2000
local_epochs = 5
num_train = 500
num_test = 1000

num = 100
s0 = [0]
n_clients = 4
spaces  = ['PowerSeries', 'Chebyshev', 'GRF', 'GRF_KL']
loss1 = []
loss2 = []
loss3 = []

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


    client_nets = []
    client_models = []
    for i in range(n_clients):
        client_nets.append(dde.nn.DeepONetCartesianProd([m, 40, 40], [dim_x, 40, 40], "relu", "Glorot normal"))
        client_data = dde.data.TripleCartesianProd(X_train=X_train_clients[i], y_train=y_train_clients[i], X_test=X_test, y_test=y_test)
        model = dde.Model(client_data, client_nets[i])
        client_models.append(model)
        client_models[i].compile("adam", lr=1e-3, metrics=["l2 relative error"])

    losshistory1, train_state = client_models[0].train(iterations = local_epochs*global_epochs)
    losshistory2, train_state = client_models[1].train(iterations = local_epochs*global_epochs)
    losshistory3, train_state = client_models[2].train(iterations = local_epochs*global_epochs)
    loss1.append(losshistory1.metrics_test[-1])
    loss2.append(losshistory2.metrics_test[-1])
    loss3.append(losshistory3.metrics_test[-1])

np.savez("antiderivative52k_extp_{}clients.npz".format(n_clients), loss1 = loss1, loss2=loss2, loss3 = loss3)
