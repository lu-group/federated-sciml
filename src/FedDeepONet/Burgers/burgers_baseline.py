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
    return solve_Burgers(XMAX=1, TMAX=1, NX=101, NT=10001, NU=0.1, u0=sensor_values)


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
T = 1
num_train = 200
num_test = 500
num = 100

loss = []

for i in range(5):

    # Load dataset
    # train_data = np.load("Burgers_traindata200_0.6.npz", allow_pickle=True)
    # test_data = np.load("Burgers_testdata500_0.6.npz", allow_pickle=True)

    # X_train, y_train = (train_data["X_train0"].astype(np.float32), train_data["X_train1"].astype(np.float32)), train_data["y_train"].astype(np.float32)
    # X_test, y_test = (test_data["X_train0"].astype(np.float32), test_data["X_train1"].astype(np.float32)), test_data["y_train"].astype(np.float32)

    space = Chebyshev(N=10, coeff_range=[-1,1], first_coeff_range=[-1, 1], direction='forward')
    X_train0, X_train1, y_train= generator_burgers(eval_s, space, T, m, num_train, num)
    X_train = (X_train0.astype(np.float32), X_train1.astype(np.float32))
    y_train = y_train.astype(np.float32)
    X_test0, X_test1, y_test = generator_burgers(eval_s, space, T, m, num_test, num)
    X_test = (X_test0.astype(np.float32), X_test1.astype(np.float32))
    y_test = y_test.astype(np.float32)

    data = dde.data.TripleCartesianProd(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    net = dde.nn.DeepONetCartesianProd(
        [m, 40, 40], [dim_x, 40, 40],
        "relu",
        "Glorot normal",
    )

    # Define a Model
    model = dde.Model(data, net)

    # Compile and Train
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(iterations=30000)
    loss.append(losshistory.metrics_test[-1])

np.savez("Burgers_baseline.npz", loss=loss)
