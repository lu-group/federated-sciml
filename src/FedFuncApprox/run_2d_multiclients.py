import numpy as np
import copy
import deepxde as dde
import skopt

# schaffer function
def f(x,y):
    return 0.5+(np.sin(4*x**2+4*y**2)**2-0.5)/(1+0.001*(4*x**2+4*y**2))**2

def create_subblocks(n, domain):
    '''
    return a list of sub-blocks, where each sub-block has an equal size
    '''
    domain_len_x = domain[1][0] - domain[0][0]
    domain_len_y = domain[1][1] - domain[0][1]
    subblock_len_x = domain_len_x / n
    subblock_len_y = domain_len_y / n
    subblock_list = []
    
    for i in range(n):
        for j in range(n):
            subblock_start = (domain[0][0] + i*subblock_len_x, domain[0][1] + j*subblock_len_y)
            subblock_end = (domain[0][0] + (i+1)*subblock_len_x, domain[0][1] + (j+1)*subblock_len_y)
            subblock_list.append((subblock_start, subblock_end))
    
    return subblock_list


def quasirandom(space, n_samples):
    '''
    space: 2D domain
    '''
    # space = [(0.0, 1.0), (0.0, 1.0)]
    sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    return np.array(sampler.generate(space, n_samples))


def data_generation_Hammersly(piece, ntrain, interval, n_clients):
    '''
    ntrain: num of sample points in total per axis
    piece: the number of cut on each axis
    interval should be the form ((0, 0), (1, 1))
    This method for now only supports 2 clients
    '''
    n_subblocks = piece + 1
    num = (ntrain// n_subblocks)**2
    subblocks = create_subblocks(n_subblocks, interval)
    
    sample_data = []
    for i in range(n_subblocks**2):
        space = [(subblocks[i][0][0],subblocks[i][1][0]),(subblocks[i][0][1],subblocks[i][1][1])]
        sample_data.append(quasirandom(space,num))
    
    datalist = []
    for i in range(n_clients):
        datalist.append([])
    
    for i in range(n_subblocks):  
        for j in range(n_subblocks): 
            client_index = (i+j) % n_clients  # calculate the client index
            data_point = np.array(sample_data[i*n_subblocks+j])  # extract the data point
            datalist[client_index].append(data_point)  # add the data point to the corresponding client's list
    
    for i in range(n_clients):
        datalist[i] = np.array(datalist[i]).reshape(-1,2)

    return datalist    


# define hyperparameters
n_clients = 3
local_epochs = 5
global_epochs = 3000
layer_size = [2] + [64] * 3 + [1]
activation = "relu"
initializer = "Glorot uniform"

n_pieces = [1,2,3,4,5,6,7,8,9,10,25]

# generate test data
x1, x2 = np.meshgrid(np.linspace(0,1,50), np.linspace(0, 1, 50))
y = f(x1, x2)
X_test = np.hstack((x1.reshape(-1,1),x2.reshape(-1,1)))
y_test = y.reshape(-1,1)

for iter in range(5):

    loss = []
    wd_fc1 = []
    wd_fc2 = []
    wd_fc3 = []
    wd_fc4 = []
    for n_cut in n_pieces:
        # generate train data
        subdata_x = data_generation_Hammersly(n_cut, 25 * n_clients, ((0,0),(1,1)), n_clients)
        subdata_y = []
        for i in range(len(subdata_x)):
            subdata_y.append(f(subdata_x[i][:,0],subdata_x[i][:,1]).reshape(-1,1))
        
        X_train = np.concatenate(subdata_x, axis=0)
        y_train = np.concatenate(subdata_y, axis=0)

        # directly define server as a FNN
        server_net = dde.nn.FNN(
            layer_size,
            activation,
            initializer,
        )

        data = dde.data.DataSet(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        server_model = dde.Model(data, server_net)
        server_model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
        
        ### Define a centralized training comparison group
        central_net = dde.nn.FNN(
            layer_size,
            activation,
            initializer,
        )
        Central = dde.Model(data, central_net)
        Central.compile("adam", lr=1e-3, metrics=["l2 relative error"])

        # Block version implementation
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.FNN(layer_size,activation,initializer))
            client_data = dde.data.DataSet(X_train=subdata_x[i], y_train=subdata_y[i], X_test=X_test, y_test=y_test)
            model = dde.Model(client_data, client_nets[i])
            client_models.append(model)
            client_models[i].compile("adam", lr=1e-3, metrics=["l2 relative error"])

        
        for i in range(n_clients):
            client_models[i].compile("adam", lr=1e-3, metrics=["l2 relative error"])

        # train
        l2 = []
        fc1 = []
        fc2 = []
        fc3 = []
        fc4 = []
        for i in range(global_epochs):
            ### Train the central comparative group
            Central.train(iterations = local_epochs)
            
            ### Train the federated decentralized group
            for j in range(len(client_models)):
                train_state = client_models[j].train(iterations=local_epochs) 

            # aggregate
            combined_dict = copy.deepcopy(client_models[0].state_dict())
            for (k1,v1), (k2,v2), (k3, v3)in zip(client_models[0].state_dict().items(), 
                                                        client_models[1].state_dict().items(),
                                                        client_models[2].state_dict().items(),
                                                        ):
                combined_dict[k1] = (v1+v2+v3)/3

            # broadcast
            for j in range(len(client_models)):
                client_nets[j].load_state_dict(combined_dict)
            
            # (optimal) test and calculate error
            if i==0 or (i+1)%1000 == 0:
                server_net.load_state_dict(combined_dict)
                y_pred = server_model.predict(X_test) 
                error = dde.metrics.l2_relative_error(y_test, y_pred)
                l2.append(error)

                ## Compute the weight divergence
                WSGD = copy.deepcopy(Central.net.state_dict())
                Wfedavg = copy.deepcopy(server_model.net.state_dict()) 
            
                fc1.append(np.linalg.norm(Wfedavg['linears.0.weight'].cpu().numpy() - WSGD['linears.0.weight'].cpu().numpy()) / np.linalg.norm(WSGD['linears.0.weight'].cpu().numpy()))
                fc2.append(np.linalg.norm(Wfedavg['linears.1.weight'].cpu().numpy() - WSGD['linears.1.weight'].cpu().numpy()) / np.linalg.norm(WSGD['linears.1.weight'].cpu().numpy()))
                fc3.append(np.linalg.norm(Wfedavg['linears.2.weight'].cpu().numpy() - WSGD['linears.2.weight'].cpu().numpy()) / np.linalg.norm(WSGD['linears.2.weight'].cpu().numpy()))
                fc4.append(np.linalg.norm(Wfedavg['linears.3.weight'].cpu().numpy() - WSGD['linears.3.weight'].cpu().numpy()) / np.linalg.norm(WSGD['linears.3.weight'].cpu().numpy()))

        loss.append(l2)
        wd_fc1.append(fc1)
        wd_fc2.append(fc2)
        wd_fc3.append(fc3)
        wd_fc4.append(fc4)

    np.savez(f"schaffer{iter}_{n_clients}clients.npz", loss = loss)
    np.savez(f"schaffer{iter}_{n_clients}clients_wd.npz", fc1 = wd_fc1, fc2 = wd_fc2, fc3 = wd_fc3, fc4 = wd_fc4)
