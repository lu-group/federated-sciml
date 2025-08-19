> Under Construction

# Federated scientific machine learning for approximating functions and solving differential equations with data heterogeneity

The data and code for the paper H. Zhang, L. Liu, K. Weng, & L. Lu. [Federated scientific machine learning for approximating functions and solving differential equations with data heterogeneity.](https://ieeexplore.ieee.org/document/11053778) IEEE Transactions on Neural Networks and Learning Systems, 2025. 

## Datasets

Data generation scripts are available in the [data](data) folder:

- [Antiderivative](data/data_gen_antid_100.py)
- [Burgers equation](data/data_gen_burgers_101_101.py)
- [Diffusion-reaction equation](data/data_gen_dr_101_101.py)
- [Lid-driven cavity flow](data/data_gen_cavity.py)

Also you may check the helper functions in [data_assignment](data/data_assignment.py) file for data generation for federated learning setting for 1D and 2D problem. Details are shown in Sec.II in the paper.

## Code

All codes are provided in the folder [src](src), including training centralized baseline models, federated models and extrapolation models. The code depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde).


### FL for Function Approximation

- [1D Gramacy&Lee function](src/FedFuncApprox/run_1d.py)
- [2D Schaffer function](src/FedFuncApprox/2d_multiclients.py)

### FedPINN

- [Poisson equation](src/FedPINN/Poisson)
- [Helmholtz equation](src/FedPINN/run_fedpinn_helmholtz.py)
- [Allen Cahn equation](src/FedPINN/run_fedpinn_allencahn.py)
- [Multiclient Allen Cahn](src/FedPINN/Multi_Allen_Cahn)
- [Inverse Navier-Stokes equation](src/FedPINN/Inverse_NS/)
- [Inverse Diffusion-reaction equation](src/FedPINN/Inverse_dr/)

### FedDeepONet

- [Antiderivative](src/FedDeepONet/Antiderivative/)
- [Burgers equation](src/FedDeepONet/Burgers/)
- [Diffusion-reaction equation](src/FedDeepONet/Diffusion_reaction/)
- [Lid-driven cavity flow](src/FedDeepONet/Cavity_flow/)
  
## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@ARTICLE{11053778,
  author   = {Zhang, Handi and Liu, Langchen and Weng, Kangyu and Lu, Lu},
  journal  = {IEEE Transactions on Neural Networks and Learning Systems}, 
  title    = {Federated Scientific Machine Learning for Approximating Functions and Solving Differential Equations With Data Heterogeneity}, 
  year     = {2025},
  volume   = {},
  number   = {},
  pages    = {1-14},
  keywords = {Data models;Mathematical models;Training;Servers;Data privacy;Distributed databases;Neural networks;Machine learning;Function approximation;Numerical models;Data heterogeneity;federated learning (FL);function approximation;operator learning;physics-informed neural networks (PINNs);scientific machine learning (SciML)},
  doi      = {10.1109/TNNLS.2025.3580409}
}

```

## Question

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
