import numpy as np
import deepxde as dde
from deepxde.backend import torch
from scipy.io import loadmat
import skopt
from pathos.pools import ProcessPool
from scipy import interpolate
from scipy.integrate import solve_ivp
from sklearn import gaussian_process as gp
from spaces import GRF
import config
import copy

def generator_ode(ode, space, T, m, n, num, dim_x):
    """Generate `n` random feature vectors.
    """
    # n: number of test or train
    # num: number of point chosen

    print("Generating operator data...", flush=True)
    features = space.random(n)
    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values = space.eval_u(features, sensors)
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

def data_1d(X,y, piece, n_clients):

    # sort the array based on the norm of each row
    norms = np.linalg.norm(X, axis=1)
    sorted_x = X[norms.argsort()]
    sorted_y = y[norms.argsort()]
                                                                            
    datalen = len(y)
    quotient =  datalen // (piece * n_clients)
    residue = datalen % (piece * n_clients)

    ### each client will get quotient * piece data if ignoring the residue
    data_listx = []
    data_listy = []

    for j in range(n_clients):
        tempx = []
        tempy = []
        resx = []
        resy = []

        for i in range(piece):
            tempx.append(sorted_x[(i*n_clients+j)*quotient:(i*n_clients+j+1)*quotient])
            tempy.append(sorted_y[(i*n_clients+j)*quotient:(i*n_clients+j+1)*quotient])

        extra = 0
        while extra * n_clients + j < residue:
            resx.append(sorted_x[quotient*piece*n_clients+ j + extra * n_clients])
            resy.append(sorted_y[quotient*piece*n_clients+ j + extra * n_clients])
            extra +=1


        data_listx.append(np.vstack((np.array(tempx).reshape(-1,len(X[0])),
                                     np.array(resx).reshape(-1,len(X[0])))))
        data_listy.append(np.vstack((np.array(tempy).reshape(-1,len(y[0])),
                                     np.array(resy).reshape(-1,len(y[0])))))

    return data_listx, data_listy

m = 100
dim_x = 1
n_clients = 2
global_epochs = 2000
local_epochs = 5
l = 0.1
T = 1
num_train = 500
num_test = 1000
num = 100
s0 = [0]
delta = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# pieces = [1,2,3,5,10,15,20,50]

for iter in range(5):

    loss1 = []
    loss2 = []
    
    for i in range(len(delta)):
        
        space1 = GRF(1, length_scale=l, N=1000, interp="cubic")
        space2 = GRF(1, length_scale=l+i, N=1000, interp="cubic")

        # federated dataset
        X_train0_client1, X_train1_client1, y_train_client1 = generator_ode(ode, space1, T, m, num_train//n_clients, num, dim_x)
        X_train0_client2, X_train1_client2, y_train_client2 = generator_ode(ode, space2, T, m, num_train//n_clients, num, dim_x)
        
        X_train_client1 = (X_train0_client1.astype(np.float32), X_train1_client1.astype(np.float32))
        y_train_client1 = y_train_client1.astype(np.float32)
        X_train_client2 = (X_train0_client2.astype(np.float32), X_train1_client2.astype(np.float32))
        y_train_client2 = y_train_client2.astype(np.float32)
        X_train_clients = (X_train_client1,X_train_client2)
        y_train_clients = (y_train_client1, y_train_client2)


        # central dataset
        X_train0_central = np.vstack((X_train0_client1, X_train0_client2))
        X_train1_central = X_train1_client1 # location points are the same
        y_train_central = np.vstack((y_train_client1, y_train_client2))

        X_test0_client1, X_test1_client1, y_test_client1 = generator_ode(ode, space1, T, m, num_test//n_clients, num, dim_x)
        X_test0_client2, X_test1_client2, y_test_client2 = generator_ode(ode, space2, T, m, num_test//n_clients, num, dim_x)

        X_test0 = np.vstack((X_test0_client1, X_test0_client2))
        X_test1 = X_test1_client1
        y_test = np.vstack((y_test_client1, y_test_client2))

        X_train = (X_train0_central.astype(np.float32), X_train1_central.astype(np.float32))
        y_train = y_train_central.astype(np.float32)
        X_test= (X_test0.astype(np.float32), X_test1.astype(np.float32))
        y_test = y_test.astype(np.float32)

    # for piece in pieces:
    #     # Load dataset
    #     X_train0 = np.load("traindata500_0.1.npz", allow_pickle=True)['X_train0']
    #     X_train1 = np.load("traindata500_0.1.npz", allow_pickle=True)['X_train1']
    #     y_train = np.load("traindata500_0.1.npz", allow_pickle=True)['y_train']
    #     X_test0 = np.load("testdata1000_0.1.npz", allow_pickle=True)['X_test0']
    #     X_test1 = np.load("testdata1000_0.1.npz", allow_pickle=True)['X_test1']
    #     y_test = np.load("testdata1000_0.1.npz", allow_pickle=True)['y_test']

    #     # centralized data
    #     X_train = (X_train0.astype(np.float32), X_train1.astype(np.float32))
    #     y_train = y_train.astype(np.float32)
    #     X_test= (X_test0.astype(np.float32), X_test1.astype(np.float32))
    #     y_test = y_test.astype(np.float32)
        
    #     # federated data
    #     X_train0_client1, X_train0_client2 =  data_1d(X_train0,y_train, piece, 2)[0]
    #     y_train_client1, y_train_client2 =  data_1d(X_train0,y_train, piece, 2)[1]
    #     X_train_client1 =  (X_train0_client1.astype(np.float32), X_train1.astype(np.float32))
    #     y_train_client1 = y_train_client1.astype(np.float32)
    #     X_train_client2 =  (X_train0_client2.astype(np.float32), X_train1.astype(np.float32))
    #     y_train_client2 = y_train_client2.astype(np.float32)
    #     X_train_clients =  (X_train_client1, X_train_client2)
    #     y_train_clients = (y_train_client1, y_train_client2)
        
        # define a list of clients as FNNs with same architecture as server
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
        loss1.append(losshistory1.metrics_test[-1])
        loss2.append(losshistory2.metrics_test[-1])

       
    np.savez("antiderivative{}_local5_2k_extp(delta).npz".format(iter), loss1 = loss1, loss2=loss2)
