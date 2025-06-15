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

    loss = []
    baseline_loss = []
    wd_fc1 = []
    wd_fc2 = []
    wd_fc3 = []
    wd_fc4 = []
    wd_fc5 = []
    wd_fc6 = []

    for d in delta:
        
        space1 = GRF(1, length_scale=l, N=1000, interp="cubic")
        space2 = GRF(1, length_scale=l+d, N=1000, interp="cubic")

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

        # # Load dataset
        # X_train0 = np.load("traindata500_0.1.npz", allow_pickle=True)['X_train0']
        # X_train1 = np.load("traindata500_0.1.npz", allow_pickle=True)['X_train1']
        # y_train = np.load("traindata500_0.1.npz", allow_pickle=True)['y_train']
        # X_test0 = np.load("testdata1000_0.1.npz", allow_pickle=True)['X_test0']
        # X_test1 = np.load("testdata1000_0.1.npz", allow_pickle=True)['X_test1']
        # y_test = np.load("testdata1000_0.1.npz", allow_pickle=True)['y_test']

        # # centralized data
        # X_train = (X_train0.astype(np.float32), X_train1.astype(np.float32))
        # y_train = y_train.astype(np.float32)
        # X_test= (X_test0.astype(np.float32), X_test1.astype(np.float32))
        # y_test = y_test.astype(np.float32)
        
        # # federated data
        # X_train0_client1, X_train0_client2 =  data_1d(X_train0,y_train, piece, 2)[0]
        # y_train_client1, y_train_client2 =  data_1d(X_train0,y_train, piece, 2)[1]
        # X_train_client1 =  (X_train0_client1.astype(np.float32), X_train1.astype(np.float32))
        # y_train_client1 = y_train_client1.astype(np.float32)
        # X_train_client2 =  (X_train0_client2.astype(np.float32), X_train1.astype(np.float32))
        # y_train_client2 = y_train_client2.astype(np.float32)
        # X_train_clients =  (X_train_client1, X_train_client2)
        # y_train_clients = (y_train_client1, y_train_client2)

        # Server model
        server_net = dde.nn.DeepONetCartesianProd([m, 40, 40], [dim_x, 40, 40], "relu", "Glorot normal")
        data = dde.data.TripleCartesianProd(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        server_model = dde.Model(data, server_net)
        server_model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
        
        ### Define a centralized training comparison group
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

        # train
        l2 = []
        l2_baseline = []
        fc1 = []
        fc2 = []
        fc3 = []
        fc4 = []

        for i in range(global_epochs):
            # Train the central model
            Central.train(iterations = local_epochs)
            
            # Train the federated models
            for j in range(len(client_models)):
                train_state = client_models[j].train(iterations=local_epochs) 

            # aggregate
            combined_dict = copy.deepcopy(client_models[0].state_dict())
            for (k1,v1), (k2,v2) in zip(client_models[0].state_dict().items(), client_models[1].state_dict().items()):
                combined_dict[k1] = (v1+v2)/2

            # broadcast
            for j in range(len(client_models)):
                client_nets[j].load_state_dict(combined_dict)
            
            # (optimal) test and calculate error
            if i==0 or (i+1)%1000 == 0:
                server_net.load_state_dict(combined_dict)
                y_pred = server_model.predict(X_test) 
                error = dde.metrics.l2_relative_error(y_test, y_pred)
                l2.append(error)
                y_pred_baseline = Central.predict(X_test) 
                error = dde.metrics.l2_relative_error(y_test, y_pred_baseline)
                l2_baseline.append(error)

                ## Compute the weight divergence
                WSGD = copy.deepcopy(Central.net.state_dict())
                Wfedavg = copy.deepcopy(server_model.net.state_dict())
            
                fc1.append(np.linalg.norm(Wfedavg['branch.linears.0.weight'].cpu().numpy() - WSGD['branch.linears.0.weight'].cpu().numpy()) / np.linalg.norm(WSGD['branch.linears.0.weight'].cpu().numpy()))
                fc2.append(np.linalg.norm(Wfedavg['branch.linears.1.weight'].cpu().numpy() - WSGD['branch.linears.1.weight'].cpu().numpy()) / np.linalg.norm(WSGD['branch.linears.1.weight'].cpu().numpy()))
                fc3.append(np.linalg.norm(Wfedavg['trunk.linears.0.weight'].cpu().numpy() - WSGD['trunk.linears.0.weight'].cpu().numpy()) / np.linalg.norm(WSGD['trunk.linears.0.weight'].cpu().numpy()))
                fc4.append(np.linalg.norm(Wfedavg['trunk.linears.1.weight'].cpu().numpy() - WSGD['trunk.linears.1.weight'].cpu().numpy()) / np.linalg.norm(WSGD['trunk.linears.1.weight'].cpu().numpy()))
                # fc3.append(np.linalg.norm(Wfedavg['trunk.linears.0.weight'].cpu().numpy() - WSGD['trunk.linears.0.weight'].cpu().numpy()) / np.linalg.norm(WSGD['trunk.linears.0.weight'].cpu().numpy()))
                # fc4.append(np.linalg.norm(Wfedavg['trunk.linears.1.weight'].cpu().numpy() - WSGD['trunk.linears.1.weight'].cpu().numpy()) / np.linalg.norm(WSGD['trunk.linears.1.weight'].cpu().numpy()))
        
        loss.append(l2)
        baseline_loss.append(l2_baseline)
        wd_fc1.append(fc1)
        wd_fc2.append(fc2)
        wd_fc3.append(fc3)
        wd_fc4.append(fc4)
        # wd_fc5.append(fc5)
        # wd_fc6.append(fc6)

    np.savez("antiderivative{}_local5_2k(delta).npz".format(iter), loss = loss)
    np.savez("antiderivative{}_local5_2k_baseline(delta).npz".format(iter), loss = baseline_loss)
    np.savez("antiderivative{}_local5_2k_wd(delta).npz".format(iter), fc1 = wd_fc1, fc2 = wd_fc2, fc3 = wd_fc3, fc4 = wd_fc4)

