import numpy as np
import sklearn.gaussian_process.kernels as kernels
from src.bayesian_optimization import bayesian_optimization
from src.benchmark_functions_2D import *

# Set seed
#np.random.seed(0)

# Benchmark Function
fun = GoldsteinPrice()
domain = fun.domain
obj_fun = lambda x: -1*fun.function(x)
arg_max = fun.arg_min

# Communication network
num_agents = 1
N = np.eye(3)
N[0,1] = N[1,0] = N[1,2] = N[2,1] = 1

# Bayesian optimization object
BO = bayesian_optimization( objective = obj_fun,
                            domain = domain,
                            arg_max = arg_max,
                            n_workers = num_agents,
                            network = None,
                            kernel = kernels.RBF(length_scale_bounds=(10**(-1), 10000)),
                            acquisition_function = 'ei',
                            stochastic_policy = False,
                            regularization = None,
                            regularization_strength = 0.01,
                            grid_density = 30)

# Optimize
BO.optimize(n_iters = 100, n_runs = 10, n_pre_samples = 3, random_search = 1000, plot = 10)
for a in range(BO.n_workers):
    print("Predicted max {}: {}".format(a, BO.pre_max[a]))
