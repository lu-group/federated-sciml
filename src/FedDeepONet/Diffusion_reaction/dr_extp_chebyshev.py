import numpy as np
import deepxde as dde
from deepxde.backend import torch
from scipy.io import loadmat
import skopt
from pathos.pools import ProcessPool
from scipy import interpolate
from scipy.integrate import solve_ivp
from sklearn import gaussian_process as gp
# from spaces import GRF
import config
import copy

import copy
from ADR_solver import solve_ADR

T = 1
D = 0.01
k = 0.01
Nt = 101
num_train = 500
num_test = 1000
m = 101
l = 0.1


def eval_s(sensor_value):
    """Compute s(x, t) over m * Nt points for a `sensor_value` of `u`.
    """
    return solve_ADR(
        0,
        1,
        0,
        T,
        lambda x: D * np.ones_like(x),
        lambda x: np.zeros_like(x),
        lambda u: k * u ** 2,
        lambda u: 2 * k * u,
        lambda x, t: np.tile(sensor_value[:, None], (1, len(t))),
        lambda x: np.zeros_like(x),
        len(sensor_value),
        Nt,
    )[2]


def generator_dr(space, T, m,n):
    """Generate `n` random feature vectors.
    """

    print("Generating operator data...", flush=True)
    features = space.random(n)
    sensors = np.linspace(0, 1, num=m)[:, None]
    sensor_values = space.eval_batch(features, sensors)
    s_values = list(map(eval_s, sensor_values))
    for j in range(len(s_values)):
        s_values[j] = np.concatenate(s_values[j])
    s_values = np.array(s_values)
    x = np.linspace(0, 1, m)
    t = np.linspace(0, T, Nt)
    xt = np.array([[a, b] for a in x for b in t])

    return sensor_values, xt, s_values


class Chebyshev():
    r"""Chebyshev polynomial.

    p(x) = \sum_{i=0}^{N-1} a_i T_i(x),
    where T_i is Chebyshev polynomial of the first kind.
    Note: The domain of x is scaled from [-1, 1] to [0, 1].

    Args:
        N (int)
        coeff_range (float): The coefficients a_i are randomly sampled from this range.
        first_coeff_range (list): The coefficients a_1 are randomly sampled from this piecewise contiuous range.
        direction: order of terms to consider. If inverse, then last N terms are non-zero and the rest are zero.
    """

    def __init__(self, N=100, coeff_range=[-1,1], first_coeff_range=[-1, 1], direction='forward'):
        self.N = N
        self.coeff_range = coeff_range
        self.first_coeff_range = first_coeff_range
        self.direction = direction

    def random(self, size):
        if self.direction == 'forward':
            M = self.coeff_range[-1]
            return 2 * M * np.random.rand(size, self.N) - M
        elif self.direction == 'inverse':
            return np.concatenate((np.zeros((size, 10-self.N)), np.random.uniform(low=self.coeff_range[0], high=self.coeff_range[1], size=(size, self.N))), axis=1)
        else:
            raise ValueError("Invalid direction. Direction must be 'forward' or 'inverse'.")
    
    def eval_one(self, feature, x):
        return np.polynomial.chebyshev.chebval(2 * x - 1, feature)

    def eval_batch(self, features, xs):
        return np.polynomial.chebyshev.chebval(2 * np.ravel(xs) - 1, features.T)

def create_subinterval(piece, n_clients, interval):
    '''
    return lists of subintervval endpoint
    '''
    datalen = interval[1] - interval[0]
    quotient =  datalen / (piece * n_clients)

    sub_interval_list = []

    for j in range(n_clients):
        temp = []

        for i in range(piece):
            temp.append([(i*n_clients+j)*quotient+interval[0], (i*n_clients+j+1)*quotient+interval[0]])

        sub_interval_list.append(temp)

    return sub_interval_list

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

m = 101
dim_x = 2
n_clients = 2
global_epochs = 10000
local_epochs = 5
n_pieces = [1,2,3,4,5,6,7,8,9,10]

for iter in range(5):

    loss1 = []
    loss2 = []

    for i in range(len(n_pieces)):
        space1 = Chebyshev(N=n_pieces[i], coeff_range=[-1,1], first_coeff_range=[-1, 1], direction='forward')
        space2 = Chebyshev(N=n_pieces[i], coeff_range=[-1,1], first_coeff_range=[-1, 1], direction='inverse')

        # federated dataset
        X_train0_client1, X_train1_client1, y_train_client1 = generator_dr(space1, T, m, num_train//n_clients)
        X_train0_client2, X_train1_client2, y_train_client2 = generator_dr(space2, T, m, num_train//n_clients)
        
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
        
        space = Chebyshev(N=10, coeff_range=[-1,1], first_coeff_range=[-1, 1], direction='forward')
        X_test0, X_test1, y_test = generator_dr(space, T, m, num_test//n_clients)

        X_train = (X_train0_central.astype(np.float32), X_train1_central.astype(np.float32))
        y_train = y_train_central.astype(np.float32)
        X_test= (X_test0.astype(np.float32), X_test1.astype(np.float32))
        y_test = y_test.astype(np.float32)
        
        # define a list of clients as FNNs with same architecture as server
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.DeepONetCartesianProd([m, 100, 100, 100], [dim_x, 100, 100, 100], "relu", "Glorot normal"))
            client_data = dde.data.TripleCartesianProd(X_train=X_train_clients[i], y_train=y_train_clients[i], X_test=X_test, y_test=y_test)
            model = dde.Model(client_data, client_nets[i])
            client_models.append(model)
            client_models[i].compile("adam", lr=1e-3, metrics=["l2 relative error"])

        # train
        losshistory1, train_state = client_models[0].train(iterations = local_epochs*global_epochs)
        losshistory2, train_state = client_models[1].train(iterations = local_epochs*global_epochs)
        loss1.append(losshistory1.metrics_test[-1])
        loss2.append(losshistory2.metrics_test[-1])

       
    np.savez("dr{}_chebyshev_extp.npz".format(iter), loss1 = loss1, loss2=loss2)
