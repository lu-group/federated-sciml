import numpy as np
import deepxde as dde
from deepxde.backend import torch
import math
import os
import warnings
from datetime import datetime
from scipy.ndimage import zoom
import random
import copy

def solve_Burgers(x_range, NT, DT, NU, u0, shock_wave_threshold=20):
    r"""Returns the velocity field and distance for 1D non-linear Burgers equation, XMIN = 0, TMIN=0
    Use FTCS scheme:
        (u_i_new - u_i_old) / dt = - u_i_old * (u_plus_old - u_minus_old) / (2*dx) 
        + nu * (u_plus_old - 2*u_i_old + u_minus_old) / (dx**2)
    In markdown \frac{u_i^{(t+1)}-u_i^{(t)}}{\Delta t}+u^{(t)}_i\frac{u_{i+1}^{(t)} -u_{i-1}^{(t)} }{2\Delta x}
                =\nu \frac{u_{i+1}^{(t)}-2u_{i}^{(t)}+u_{i-1}^{(t)}}{\Delta x^2}
    """
    if NU == 0:
        raise ValueError("Inviscid Burgers' equation is not supported.")
    NX = len(u0)
    DX = x_range / (NX - 1)

    u0_max = np.max(np.abs(u0)).item()
    threshold_1 = 2*NU / (u0_max)**2
    threshold_2 = (DX**2) / (2*NU)
    if DT > threshold_1 or DT > threshold_2:
        # Necessary condition for stability
        min_DT = min(threshold_1, threshold_2)
        raise ValueError(f"Unstable solution. The maximum DT should be {min_DT}, thresholds are {threshold_1} and {threshold_2}.")

    # Initialise data structures
    u = np.zeros((NX - 1, NT)).astype(np.float32)

    # Initial conditions
    u[:, 0] = u0[:-1]

    # Periodic boundary conditions
    I = np.eye(NX - 1)
    I1 = np.roll(I, 1, axis=0)
    I2 = np.roll(I, -1, axis=0)
    A = I2 - I1
    B = I1 + I2 - 2 * I

    shock_wave_flag = False # Most runtime warnings are due to the shock wave. 

    for n in range(0, NT - 1):
        u[:, n + 1] = u[:, n] - (DT / (2 * DX)) * np.dot(A, u[:, n]) * u[:, n] + NU * (DT / DX ** 2) * np.dot(B, u[:, n])
        if np.max(np.abs(u[:, n + 1])) > shock_wave_threshold:
            shock_wave_flag = True
            print(f"Shock wave detected at time step {n}. Computing is stopped.")
            break # Stop the computation if the shock wave is detected. The rest of u[:, n+1:] will be zeros.

    u = np.concatenate([u, u[0:1, :]], axis=0)
    return u, shock_wave_flag # u.shape = (NX, NT)

def complete_fourier_initial_conditions(num_steps, x_grid, max_wavesnum, num_samples_per_step, mode, pi_phase, pooled=False):
    """
    Args:
        pi_phase: Should be in [0, 0.5].
    """
    if mode == 'cos':
        base_funcs = [np.cos(wavenum * 2 * math.pi / (x_grid[-1] - x_grid[0]) * x_grid) for wavenum in range(1, max_wavesnum + 1)]
        phase_funcs = [np.cos( wavenum * 2 * math.pi / (x_grid[-1] - x_grid[0]) * x_grid + math.pi*pi_phase ) for wavenum in range(1, max_wavesnum + 1)]
    elif mode == 'sin':
        base_funcs = [np.sin(wavenum * 2 * math.pi / (x_grid[-1] - x_grid[0]) * x_grid) for wavenum in range(1, max_wavesnum + 1)]
        phase_funcs = [np.sin( wavenum * 2 * math.pi / (x_grid[-1] - x_grid[0]) * x_grid + math.pi*pi_phase ) for wavenum in range(1, max_wavesnum + 1)]     
    elif mode == 'both':
        base_funcs = [np.sin(wavenum * 2 * math.pi / (x_grid[-1] - x_grid[0]) * x_grid) for wavenum in range(1, max_wavesnum + 1)]
        phase_funcs = [np.cos(wavenum * 2 * math.pi / (x_grid[-1] - x_grid[0]) * x_grid) for wavenum in range(1, max_wavesnum + 1)] 
        warnings.warn('Mode "both" equals Mode "sin" or "cos" with pi_phase=0.5.', UserWarning) if pi_phase != 0.5 else None
    else:
        raise ValueError("mode should be 'sin', 'cos' or 'both'.")
    if not 0<=pi_phase<=0.5:
        warnings.warn("The pi phase is not in [0, 0.5].", UserWarning)
    # constant_func = np.ones_like(x_grid) # TODO: Decide whether to include the constant function.

    step_exponents = np.linspace(1.2, 2, num_steps)
    NX = len(x_grid)
    continual_init_conditions = np.zeros((num_steps, num_samples_per_step, NX))
    max_amplitudes_list = []
    for step in range(num_steps):
        max_amplitudes = np.array( [(1/i)**step_exponents[step] for i in range(1, max_wavesnum + 1)] )
        coefficients_base = max_amplitudes * np.random.uniform(-1, 1, (num_samples_per_step, max_wavesnum))
        coefficients_phase = max_amplitudes * np.random.uniform(-1, 1, (num_samples_per_step, max_wavesnum))
        # TODO: Decide whether to use uniform or cos(uniform).
        base_terms = np.sum(coefficients_base[:, :, np.newaxis] * np.array(base_funcs)[np.newaxis, :, :], axis=1)
        phase_terms = np.sum(coefficients_phase[:, :, np.newaxis] * np.array(phase_funcs)[np.newaxis, :, :], axis=1)
        continual_init_conditions[step] = base_terms + phase_terms

        max_amplitudes_list.append(max_amplitudes)
    if pooled:
        continual_init_conditions = continual_init_conditions.reshape(1, num_steps * num_samples_per_step, NX)

    info_txt = [f"The initial conditions are sampled from the function space spanned by Fourier basis functions with the maximum waves number {max_wavesnum}.",
                f"Both sin and cos functions are included" if mode == 'both' else f"The {mode} functions are included. And there are also {mode} functions with a phase shift of {pi_phase} pi",
                f"The maximum amplitude of the first basis is 1, and the others are decayed exponentially.",
                f"The amplitudes of each samples are then uniformly distributed in [-max_amp, max_amp]."
                f"For each time step, the decay exponents are {[f'{exp:.3f}' for exp in step_exponents]}. And the maximum amplitudes of each step are:",
                "\n".join(f"  Step {step}: " + ", ".join(f"{amp:.3f}" for amp in max_amplitudes) 
                for step, max_amplitudes in enumerate(max_amplitudes_list)),]
    info_txt.append("Datasets at different steps are pooled into a one-step dataset.") if pooled else None
    return continual_init_conditions, info_txt

def gen_client_datasets_a_step(v_matrices, NU, x_grid, t_grid, DT, num_sensors=None):
    """ Generate datasets for a client at a continual learning step. 
    Shock waves are detected and removed from the output np_fedconti_datasets.
    The datasets are aligned datasets for operator learning from 1d functions to 2d functions. 
    
    Args:
        v_matrices: 2d np.ndarray or a list of 1d np.ndarrays. Each v_vec represents an initial condition.
        NU: the viscosity of the Burgers' equation.
        x_grid: the spatial grid for the initial conditions and solutions of the Burgers' equation.
                x_grid could be coarser than v vectors in v_matrices.
        t_grid: the temporal grid for the solutions of the Burgers' equation.
        DT: used in the computing format, but not the interval of t_grid.
    
    Returns:
        initial_conditions: A 2d np.ndarray with shape (num_samples, len(x_grid)).
        solutions: A 3d np.ndarray with shape (num_samples, len(x_grid), len(t_grid)).
        shock_waves: A 3d np.ndarray with shape (num_shock_waves, len(x_grid), len(t_grid)).
    """
    XMAX = x_grid[-1]
    XMIN = x_grid[0]
    TMAX = t_grid[-1]
    NT = math.ceil(TMAX / DT) + 1
    
    solutions = []
    shock_waves = []
    for v_vec in v_matrices:
        u_matrix, shock_flag = solve_Burgers(XMAX-XMIN, NT, DT, NU, u0=v_vec)

        u_matrix = zoom(u_matrix, (len(x_grid) / u_matrix.shape[0], len(t_grid) / u_matrix.shape[1]), output=np.float32)
        assert u_matrix.shape == (len(x_grid), len(t_grid)), f"u_matrix shape is {u_matrix.shape}, but should be {(len(x_grid), len(t_grid))}."

        if shock_flag:
            shock_waves.append(u_matrix)
        else:
            solutions.append(u_matrix)

    initial_conditions = v_matrices
    if num_sensors is not None:
        initial_conditions = zoom(initial_conditions, (1, num_sensors / initial_conditions.shape[1]), output=np.float32)
        assert initial_conditions.shape[1] == num_sensors, f"initial_conditions shape is {initial_conditions.shape}, but should be {(initial_conditions.shape[0], num_sensors)}."
        

    return initial_conditions, np.array(solutions), np.array(shock_waves)


def gen_client_continual_deeponet_dataset(continual_init_conditions_dataset, name):
    print(f"Computing the {name} datasets")
    continual_deeponet_branch_inputs = []
    continual_deeponet_outputs = []
    for step, init_conditions in enumerate(continual_init_conditions_dataset):
        init_conditions, solutions, _ =  gen_client_datasets_a_step(init_conditions, NU, x_grid, t_grid, DT, num_sensors=len(x_grid))
        print(f"Process: {step+1}/{continual_init_conditions_dataset.shape[0]}")

        continual_deeponet_branch_inputs.append( init_conditions.astype(np.float32) )
        continual_deeponet_outputs.append( solutions.reshape(solutions.shape[0], -1).astype(np.float32) )

    trunk_input = np.array([(x, t) for x in x_grid for t in t_grid], dtype=np.float32)
    return continual_deeponet_branch_inputs, continual_deeponet_outputs, trunk_input


num_steps = 1
max_wavesnum = 9
NU = 0.1
num_clients = 2 
client_pi_phases = np.linspace(0,0.5,6)
# client_pi_phases = [0, 0.5]

# For final datasets.
x_grid = np.linspace(0, 1, 101)
XMIN = x_grid[0]
XMAX = x_grid[-1]
t_grid = np.linspace(0, 1, 101)

# Used in computation scheme. Could be different from x_grid and t_grid.
NX = 201
DT = 5e-6

num_samples_per_client_per_step = 200
num_samples_test = 400

n_clients = 2
global_epochs = 20000
local_epochs = 5


for iter in range(5):

    loss1 = []
    loss2 = []

    for client_pi_phase in client_pi_phases:
     
        client1_continual_init_conditions, client1_info_txt = complete_fourier_initial_conditions(num_steps, x_grid, max_wavesnum, num_samples_per_client_per_step, mode='sin', pi_phase=client_pi_phase,
                                                                                            pooled=True)
        client2_continual_init_conditions, client2_info_txt = complete_fourier_initial_conditions(num_steps, x_grid, max_wavesnum, num_samples_per_client_per_step, mode='cos', pi_phase=client_pi_phase,
                                                                                                pooled=True)
        test_continual_init_conditions, test_info_txt = complete_fourier_initial_conditions(num_steps, x_grid, max_wavesnum, int(num_samples_test/num_steps), mode='both', pi_phase=0.5, pooled=True)

        X_train0_client1, y_train_client1, X_train1_client1 = gen_client_continual_deeponet_dataset(client1_continual_init_conditions, 'client1')
        X_train0_client2, y_train_client2, X_train1_client2 = gen_client_continual_deeponet_dataset(client2_continual_init_conditions, 'client2')
        X_test0, y_test, X_test1 = gen_client_continual_deeponet_dataset(test_continual_init_conditions, 'test')

        X_train_client1 = (X_train0_client1[0].astype(np.float32), X_train1_client1.astype(np.float32))
        y_train_client1 = y_train_client1[0].astype(np.float32)
        X_train_client2 = (X_train0_client2[0].astype(np.float32), X_train1_client2.astype(np.float32))
        y_train_client2 = y_train_client2[0].astype(np.float32)

        X_train_clients = (X_train_client1,X_train_client2)
        y_train_clients = (y_train_client1, y_train_client2)
        
        X_test = (X_test0[0].astype(np.float32), X_test1.astype(np.float32))
        y_test = y_test[0].astype(np.float32)


        # define a list of clients as FNNs with same architecture as server
        client_nets = []
        client_models = []
        for i in range(n_clients):
            client_nets.append(dde.nn.DeepONetCartesianProd([101, 64, 64, 64], [2, 64, 64, 64], "relu", "Glorot normal"))
            client_data = dde.data.TripleCartesianProd(X_train=X_train_clients[i], y_train=y_train_clients[i], X_test=X_test, y_test=y_test)
            model = dde.Model(client_data, client_nets[i])
            client_models.append(model)
            client_models[i].compile("adam", lr=1e-3, metrics=["l2 relative error"])


        losshistory1, train_state = client_models[0].train(iterations = local_epochs*global_epochs)
        losshistory2, train_state = client_models[1].train(iterations = local_epochs*global_epochs)
        loss1.append(losshistory1.metrics_test[-1])
        loss2.append(losshistory2.metrics_test[-1])

        torch.save(client_models[0].state_dict(), f"extp1_model_phi{client_pi_phase}.pth")
        torch.save(client_models[1].state_dict(), f"extp2_model_phi{client_pi_phase}.pth")
    np.savez(f"burgers{iter}_extp.npz", loss1 = loss1, loss2 = loss2)
