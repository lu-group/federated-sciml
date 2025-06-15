import copy
import numpy as np
import deepxde as dde
from deepxde.backend import torch
import skopt
from scipy.io import loadmat

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


def gen_block(x_train, y_train, num_cut):
    """
    dataset: nparray
    return: data block x and y after cutting
    for 40000 data points, 1 cut, return [4,10000,2] and [4,10000,1]
    """
    datalen = len(y_train)
    sidelen = int(np.sqrt(datalen))
    blocklen = sidelen//(num_cut+1)

    data_blockx = []
    data_blocky = []
 
    # loop vertical block
    for n in range(num_cut+1):
        # loop vertical block
        for j in range(num_cut+1):
            temp_x = []
            temp_y = []
            # block_idx = (sidelen//blocklen) * n + j
            for i in range(blocklen):
                # get data inside one block
                left_idx = (n * blocklen + i) * sidelen + j * blocklen
                right_idx = (n * blocklen + i) * sidelen + (j+1) * blocklen
                # print("{}idx:".format((sidelen//blocklen) * n + j), left_idx, right_idx)
                temp_x.append([x_train[i] for i in range(left_idx,right_idx)])
                temp_y.append([y_train[i] for i in range(left_idx,right_idx)])
            # print(np.shape(temp_x), len(temp_x[0]), len(temp_x[1]))
            data_blockx.append(temp_x)
            data_blocky.append(temp_y)

    # reshape to [n_block, datalen//n_block, dim]
    mx = np.array(data_blockx).shape[:][0]
    nx = np.array(data_blockx).shape[:][-1]
    data_x = np.array(data_blockx).reshape(mx,-1,nx)

    my = np.array(data_blocky).shape[:][0]
    ny = np.array(data_blocky).shape[:][-1]
    data_y = np.array(data_blocky).reshape(my,-1,ny)
            
    return data_x, data_y


def data_assignmentX_1d(X_train, piece, n_clients):

    datalen = len(X_train)
    quotient =  datalen // (piece * n_clients)
    residue = datalen % (piece * n_clients)

    data_listx = []

    for j in range(n_clients):
        tempx = []
        resx = []

        for i in range(piece):
            tempx.append(X_train[(i*n_clients+j)*quotient:(i*n_clients+j)*quotient+quotient])

        extra = 0
        while extra * n_clients + j < residue:
            resx.append(X_train[quotient*piece*n_clients+ j + extra * n_clients])
            extra +=1

        data_listx.append(np.vstack((np.array(tempx).reshape(-1,1), np.array(resx).reshape(-1,1))))

    return data_listx


def data_assignment_2d(X_train, y_train, n_cut, n_clients):
    """This is to split the dataset recursively
    dataset: TensorDataset to split
    n: number of cut on each direciton
    clinets: a list of Client class
    """
    datalist_x = []
    datalist_y = []
    n_block = (n_cut+1)**2
    
    data_blockx, data_blocky = gen_block(X_train, y_train, n_cut)
    # can evenly split to n clients
    if n_block % n_clients == 0:
       
       for i in range(n_clients):
            tempx = []
            tempy = []
            
            # row-wise append data
            for j in range(n_cut+1):
                l_pt = (i + j) % n_clients + j * (n_cut + 1)
                r_pt = (i + j) % n_clients + (j + 1) * (n_cut + 1)
                tempx.append(data_blockx[l_pt:r_pt:n_clients].reshape(-1, len(data_blockx[0][0])))
                tempy.append(data_blocky[l_pt:r_pt:n_clients].reshape(-1, len(data_blocky[0][0])))
           
            # give client i data_x together in shape [block*data/block, 2]
            datalist_x.append(np.array(tempx).reshape(-1, len(data_blockx[0][0])))
            datalist_y.append(np.array(tempy).reshape(-1, len(data_blocky[0][0])))
            
    # concatenate the residue nunmber of blocks and resplit to assign
    else:
        rest_datax = data_blockx[-1].reshape(-1,len(data_blockx[0][0]))
        rest_datay = data_blocky[-1].reshape(-1,len(data_blocky[0][0]))
        rest_blockx, rest_blocky = gen_block(rest_datax, rest_datay,n_clients-1)
        
        for i in range(n_clients):
            tempx = np.vstack((data_blockx[i:-(n_block % n_clients):n_clients].reshape(-1, len(data_blockx[0][0])),
                                rest_blockx[i::n_clients].reshape(-1, len(rest_blockx[0][0]))))
            tempy = np.vstack((data_blocky[i:-(n_block % n_clients):n_clients].reshape(-1, len(data_blocky[0][0])),
                                rest_blocky[i::n_clients].reshape(-1, len(rest_blocky[0][0]))))
            datalist_x.append(tempx)
            datalist_y.append(tempy)

    return datalist_x, datalist_y


def data_assignment_2d_slice(interval1, interval2, num, n_cut, n_clients):
    """This is to split the dataset recursively only in x-axis
    interval1, interval2: list of intervals for x and y-axis
    n: number of cut on each direciton
    clinets: a list of Client class
    """
    quotient =  num // (n_cut * n_clients)
    residue =  num % (n_cut * n_clients)

    X = np.linspace(interval1[0],interval1[1],num)
    data_listx = []
    
    for j in range(n_clients):
        tempx = []
        resx = []

        for i in range(n_cut):
            tempx.append(X[(i*n_clients+j)*quotient : (i*n_clients+j)*quotient+quotient])

        extra = 0
        while extra * n_clients + j < residue:
            resx.append(X[quotient * n_cut * n_clients + j + extra * n_clients])
            extra +=1

        data_listx.append(np.vstack((np.array(tempx).reshape(-1,1), np.array(resx).reshape(-1,1))))

    data = []
    for i in range(n_clients):
        x11, x12 = np.meshgrid(data_listx[i], np.linspace(interval2[0], interval2[1], num))
        data.append(np.hstack((x11.reshape(-1,1), x12.reshape(-1,1))))

    return data

def quasirandom(space, n_samples):
    '''
    space: 2D domain
    '''
    # space = [(0.0, 1.0), (0.0, 1.0)]
    sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    return np.array(sampler.generate(space, n_samples))


def data_generation_Hammersly(piece, ntrain, interval):
    '''
    ntrain: num of sample points in each subblock
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
    datalist = []
    if (n_subblocks**2) % 2 == 0:
        client1 = []
        client2 = []
        for i in range(n_subblocks):  # iterate over the rows
            for j in range(n_subblocks):  # iterate over the columns
                if (i+j)%2 == 0:  # check if the sum of row and column indices is even
                    client1.append(np.array(sample_data[i*n_subblocks+j])) # add the element to client 0's list
                else:
                    client2.append(np.array(sample_data[i*n_subblocks+j]))
        datalist.append(np.array(client1).reshape(-1,2))
        datalist.append(np.array(client2).reshape(-1,2))
        # datalist.append(np.array(sample_data[l_pt:r_pt:n_clients]).reshape(-1,2))
            
    else:
        client1 = np.vstack((np.array(sample_data[:-1][0::2]).reshape(-1,2),np.array(sample_data[-1][:len(sample_data[0])//2])))
        client2 = np.vstack((np.array(sample_data[:-1][1::2]).reshape(-1,2),np.array(sample_data[-1][len(sample_data[0])//2:])))
        datalist.append(client1)
        datalist.append(client2)

    return datalist