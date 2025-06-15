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

class GRF(object):
    def __init__(self, T, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, T, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        K = K(self.x)
        self.L = np.linalg.cholesky(K + 1e-13 * np.eye(self.N))

    def random(self, n):
        """Generate `n` random feature vectors.
        """
        u = np.random.randn(self.N, n)
        return np.dot(self.L, u).T

    # def random_bounded(self, l_bc, r_bc, n):
    #     """Generate `n` random feature vectors with norms bounded by certain values.
    #     """
    #     def randn_bounded(lower, upper, shape):
    #         while True:
    #             x = np.random.randn(*shape)
    #             norm = np.linalg.norm(x)
    #             if lower <= norm <= upper:
    #                 return x

    #     # Define lower and upper bounds for L2 norm
    #     lower = l_bc
    #     upper = r_bc

    #     # Generate n arrays of size (n,) with L2 norm bounded between lower and upper
    #     u = np.array([randn_bounded(lower, upper, (self.N,)) for _ in range(n)]).T
    #     return np.dot(self.L, u).T
    
    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`.
        """
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), y)
        f = interpolate.interp1d(
            np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        if self.interp == "linear":
            return np.vstack([np.interp(sensors, np.ravel(self.x), y).T for y in ys])
        p = ProcessPool(nodes=config.processes)
        res = p.map(
            lambda y: interpolate.interp1d(
                np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
            )(sensors).T,
            ys,
        )
        return np.vstack(list(res))
    
def generator_ode(ode, space, T, m, n, num, dim_x):
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

m = 100
T = 1
num_train = 100
num_test = 1000
s0 = [0]
dim_x = 1
num = 100
space = GRF(1, length_scale=0.1, N=1000, interp="cubic")

X_train0, X_train1, y_train = generator_ode(ode, space, T, m, num_train , num, dim_x)
X_test0, X_test1, y_test = generator_ode(ode, space, T, m, num_test, num, dim_x)
np.savez("traindata{}_0.1.npz".format(num_train), X_train0=X_train0, X_train1 = X_train1, y_train=y_train)
np.savez("testdata{}_0.1.npz".format(num_test), X_test0=X_test0, X_test1 = X_test1, y_test=y_test)