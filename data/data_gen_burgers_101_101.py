import numpy as np
from spaces import GRF

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


l = 0.6
T = 1
num_train = 200
num_test = 500
m = 101
Nt = 101

print("Generating operator data...", flush=True)
space = GRF(T, length_scale=l, N=1000 * T, interp="cubic")
features = space.random(num_train)
sensors = np.linspace(0, 1, num=m)[:, None]
sensor_values = space.eval_u(features, sensors)
s = np.array(list(map(eval_s, sensor_values,)))
print(np.shape(s))
s_values = s[:, :, ::100].reshape(len(sensor_values), 101*101)
x = np.linspace(0, 1, m)
t = np.linspace(0, T, Nt)
xt = np.array([[a, b] for a in x for b in t])
print(np.shape(sensor_values), np.shape(xt), np.shape(s_values))
np.savez(f"Burgers_traindata{num_train}_{l}.npz", X_train0=sensor_values, X_train1=xt, y_train=s_values)

print("Generating operator data...", flush=True)
space = GRF(T, length_scale=l, N=1000 * T, interp="cubic")
features = space.random(num_test)
sensors = np.linspace(0, 1, num=m)[:, None]
sensor_values = space.eval_u(features, sensors)
s = np.array(list(map(eval_s, sensor_values,)))
print(np.shape(s))
s_values = s[:, :, ::100].reshape(len(sensor_values), 101*101)
x = np.linspace(0, 1, m)
t = np.linspace(0, T, Nt)
xt = np.array([[a, b] for a in x for b in t])
print(np.shape(sensor_values), np.shape(xt), np.shape(s_values))
np.savez(f"Burgers_testdata{num_test}_{l}.npz", X_train0=sensor_values, X_train1=xt, y_train=s_values)





