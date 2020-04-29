import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process.kernels as kernels
from src.bayesian_optimization import bayesian_optimization
from src.benchmark_functions import *

# Set numpy random seed
np.random.seed(0)

# Domain
domain = np.array([[-5, 5], [-5, 5]])

# Objective function
obj_fun = lambda x: (0.4 * np.exp(-((x[0]+2)**2 + (x[1]+4)**2)/4**2) + \
                    0.3 * np.exp(-((x[0]-3)**2 + (x[1]-1)**2)/2**2) + \
                    0.3 * np.exp(-((x[0]+3)**2 + (x[1]-3)**2)/5**2))

fun = Bohachevsky_1()
domain = fun.domain
obj_fun = fun.function

# Communication network
num_agents = 3
N = np.eye(3)
N[0,1] = N[1,0] = N[1,2] = N[2,1] = 1

# Bayesian optimimzation object
BO = bayesian_optimization( obj = obj_fun, domain = domain,
                            n_workers = num_agents, network = N,
                            kernel = kernels.RBF(length_scale_bounds=(1, 10000)),
                            acquisition_function = 'ei',
                            stochastic_policy = False,
                            regularization = None, l = 0.01,
                            grid_density = 30)

# Optimize
BO.optimize(n_iters = 10, n_pre_samples = 3, random_search = 1000, plot = False)
for a in range(BO.n_workers):
    print("Predicted optimum {}: {}".format(a, BO.predicted_optimum[a]))
