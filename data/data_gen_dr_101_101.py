import numpy as np
from spaces import GRF
from ADR_solver import solve_ADR

T = 1
D = 0.01
k = 0.01
# number of time points
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


print("Generating operator data...", flush=True)
space = GRF(T, length_scale=0.1, N=1000 * T, interp="cubic")
features = space.random(num_train)
sensors = np.linspace(0, 1, num=m)[:, None]
sensor_values = space.eval_u(features, sensors)
s_values = list(map(eval_s, sensor_values))
for j in range(len(s_values)):
    s_values[j] = np.concatenate(s_values[j])
s_values = np.array(s_values)
x = np.linspace(0, 1, m)
t = np.linspace(0, T, Nt)
xt = np.array([[a, b] for a in x for b in t])
np.savez(f"dr_traindata{num_train}_{l}.npz", X_train0=sensor_values, X_train1=xt, y_train=s_values)


# print("Generating operator data...", flush=True)
# features = space.random(num_test)
# sensors = np.linspace(0, 1, num=m)[:, None]
# sensor_values = space.eval_u(features, sensors)
# s_values = list(map(eval_s, sensor_values))
# for j in range(len(s_values)):
#     s_values[j] = np.concatenate(s_values[j])
# s_values = np.array(s_values)
# x = np.linspace(0, 1, m)
# t = np.linspace(0, T, Nt)
# xt = np.array([[a, b] for a in x for b in t])
# np.savez(f"dr_testdata{num_test}_{l}.npz", X_test0=sensor_values, X_test1=xt, y_test=s_values)

