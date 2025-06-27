> Under Construction

# Federated scientific machine learning for approximating functions and solving differential equations with data heterogeneity

The data and code for the paper [H. Zhang, L. Liu, K. Weng, & L. Lu. Federated scientific machine learning for approximating functions and solving differential equations with data heterogeneity](https://doi.org/10.22331/q-2025-06-04-1761)

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

- [Helmholtz equation](src/FedPINN/run_fedpinn_helmholtz.py)
- [Allen Cahn equation](src/FedPINN/run_fedpinn_allencahn.py)
- [Inverse Diffusion-reaction equation](src/FedPINN/Inverse_dr/)

### FedDeepONet

- [Antiderivative](src/FedDeepONet/Antiderivative/)
- [Burgers equation](src/FedDeepONet/Burgers/)
- [Diffusion-reaction equation](src/FedDeepONet/Diffusion_reaction/)
- [Lid-driven cavity flow](src/FedDeepONet/Cavity_flow/)
  
## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{zhang2024federated,
  title={Federated scientific machine learning for approximating functions and solving differential equations with data heterogeneity},
  author={Zhang, Handi and Liu, Langchen and Weng, Kangyu and Lu, Lu},
  journal={arXiv preprint arXiv:2410.13141},
  year={2024}
}
```

## Question

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
