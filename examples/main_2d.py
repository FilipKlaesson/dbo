import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process.kernels as kernels
from src.bayesian_optimization import bayesian_optimization
from src.benchmark_functions import *

# Set seed
np.random.seed(0)

# Benchmark Function
fun = Bohachevsky_1()
domain = fun.domain
obj_fun = lambda x: -1*fun.function(x)
arg_max = fun.arg_min

# Communication network
num_agents = 1
N = np.eye(3)
N[0,1] = N[1,0] = N[1,2] = N[2,1] = 1

# Bayesian optimization object
BO = bayesian_optimization( obj = obj_fun,
                            domain = domain,
                            arg_max = arg_max,
                            n_workers = num_agents,
                            network = None,
                            kernel = kernels.RBF(length_scale_bounds=(1, 10000)),
                            acquisition_function = 'ei',
                            stochastic_policy = False,
                            regularization = None,
                            l = 0.01,
                            grid_density = 30)

# Optimize
BO.optimize(n_iters = 10, n_runs = 2, n_pre_samples = 3, random_search = 1000, plot = True)
for a in range(BO.n_workers):
    print("Predicted max {}: {}".format(a, BO.pre_max[a]))
