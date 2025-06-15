import numpy as np
import deepxde as dde

def f(x):
    return (x + 0.5)**4 - np.sin(10 * np.pi * x) / (2*x+3)

ftrain = np.loadtxt("Gramacy&Lee(2012)_train.txt")
ftest = np.loadtxt("Gramacy&Lee(2012)_test.txt")
X_train = ftrain[:,:1]
y_train =ftrain[:,1][:,None]
X_test = ftest[:,:1]
y_test =ftest[:,1][:,None]

# define hyperparameters
n_clients = 2
local_epochs = 3
global_epochs = 5000
layer_size = [1] + [64] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
loss = []
# 1D federated
for iter in range(5):

    # directly define server as a FNN
    server_net = dde.nn.FNN(
        layer_size,
        activation,
        initializer,
    )
    data = dde.data.DataSet(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    server_model = dde.Model(data, server_net)
    server_model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    losshistory, train_state = server_model.train(epochs=local_epochs * global_epochs)
    loss.append(losshistory.metrics_test[-1])

np.savez("1D_baseline.npz", loss=loss)
