import numpy as np
import deepxde as dde
from deepxde.backend import torch
import copy

class Chebyshev():
    r"""Chebyshev polynomial.

    p(x) = \sum_{i=0}^{N-1} a_i T_i(x),
    where T_i is Chebyshev polynomial of the first kind.
    Note: The domain of x is scaled from [-1, 1] to [0, 1].

    Args:
        N (int)
        coeff_range (float): The coefficients a_i are randomly sampled from this range.
        first_coeff_range (list): The coefficients a_1 are randomly sampled from this piecewise contiuous range.
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


def solve_Burgers(XMAX, TMAX, NX, NT, NU, u0):
    """
   Returns the velocity field and distance for 1D non-linear Burgers equation, XMIN = 0, TMIN=0
   """

    # Increments
    DT = TMAX / (NT - 1)
    DX = XMAX / (NX - 1)

    # Initialise data structures
    u = np.zeros((NX, NT))
    ipos = np.zeros(NX-1, dtype=int)
    ineg = np.zeros(NX-1, dtype=int)

    # Initial conditions
    u[:-1, 0] = u0[0:-1]

    # Periodic boundary conditions
    for i in range(0, NX-1):
        ipos[i] = i + 1
        ineg[i] = i - 1

    ipos[NX - 2] = 0
    ineg[0] = NX - 2

    # Numerical solution
    for n in range(0,NT-1):
       for i in range(0,NX-1):
           u[i,n+1] = (u[i,n]-u[i,n]*(DT/(2*DX))*(u[ipos[i],n]-u[ineg[i],n])+NU*(DT/DX**2)*(u[ipos[i],n]-2*u[i,n]+u[ineg[i],n]))
    u[-1, :] = u[0, :]
    return u


def eval_s(sensor_values):
    return solve_Burgers(XMAX=1, TMAX=1, NX=101, NT=10001, NU=0.01, u0=sensor_values)


def generator_burgers(eval_s, space, T, m, n, num):
    """Generate `n` random feature vectors.
    """
    # n: number of test or train
    # num: number of point chosen
    print("Generating operator data...", flush=True)
    features = space.random(n)
    sensors = np.linspace(0, 1, num=m)[:, None]
    sensor_values = space.eval_batch(features, sensors)
    s = np.array(list(map(eval_s, sensor_values,)))
    s_values = s[:, :, ::100].reshape(len(sensor_values), 101*101)
    x = np.linspace(0, 1, m)
    t = np.linspace(0, T, Nt)
    xt = np.array([[a, b] for a in x for b in t])

    return sensor_values, xt, s_values




m = 101
Nt = 101
dim_x = 2
n_clients = 2
global_epochs = 10000
local_epochs = 5
l = 0.6
T = 1
num_train = 200
num_test = 1000
num = 100
n_pieces = [1,2,3,4,5,6,7,8,9,10]

for iter in range(5):

    loss1 = []
    loss2 = []

    for i in range(len(n_pieces)):
        
        space1 = Chebyshev(N=n_pieces[i], coeff_range=[-1,1], first_coeff_range=[-1, 1], direction='forward')
        space2 = Chebyshev(N=n_pieces[i], coeff_range=[-1,1], first_coeff_range=[-1, 1], direction='inverse')

        # federated dataset
        X_train0_client1, X_train1_client1, y_train_client1 = generator_burgers(eval_s, space1, T, m, num_train//n_clients, num)
        X_train0_client2, X_train1_client2, y_train_client2 = generator_burgers(eval_s, space2, T, m, num_train//n_clients, num)
        
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
        X_test0, X_test1, y_test = generator_burgers(eval_s, space, T, m, num_test, num)

        X_train = (X_train0_central.astype(np.float32), X_train1_central.astype(np.float32))
        y_train = y_train_central.astype(np.float32)
        X_test= (X_test0.astype(np.float32), X_test1.astype(np.float32))
        y_test = y_test.astype(np.float32)


        # Server model
        server_net = dde.nn.DeepONetCartesianProd([m, 100, 100, 100], [dim_x, 100, 100, 100], "relu", "Glorot normal")
        data = dde.data.TripleCartesianProd(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        server_model = dde.Model(data, server_net)
        server_model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
        
        ### Define a centralized training comparison group
        central_net = dde.nn.DeepONetCartesianProd([m,100, 100, 100], [dim_x, 100, 100, 100], "relu", "Glorot normal")
        central_data = dde.data.TripleCartesianProd(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        Central = dde.Model(central_data, central_net)
        Central.compile("adam", lr=1e-3, metrics=["l2 relative error"])

        # define a list of clients as FNNs with same architecture as server
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.DeepONetCartesianProd([m, 100, 100, 100], [dim_x, 100, 100, 100], "relu", "Glorot normal"))
            client_data = dde.data.TripleCartesianProd(X_train=X_train_clients[i], y_train=y_train_clients[i], X_test=X_test, y_test=y_test)
            model = dde.Model(client_data, client_nets[i])
            client_models.append(model)
            client_models[i].compile("adam", lr=1e-3, metrics=["l2 relative error"])


        losshistory1, train_state = client_models[0].train(iterations = local_epochs*global_epochs)
        losshistory2, train_state = client_models[1].train(iterations = local_epochs*global_epochs)
        loss1.append(losshistory1.metrics_test[-1])
        loss2.append(losshistory2.metrics_test[-1])

       
    np.savez("burgers{}_extp.npz".format(iter), loss1 = loss1, loss2 = loss2)
