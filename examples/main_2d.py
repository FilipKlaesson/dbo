import numpy as np
import sklearn.gaussian_process.kernels as kernels
from src.bayesian_optimization import bayesian_optimization
from src.benchmark_functions_2D import *

# Set seed
np.random.seed(0)

# Benchmark Function
fun = Ackley()
domain = fun.domain
obj_fun = lambda x: -1*fun.function(x)
arg_max = fun.arg_min

# Communication network
N = np.eye(3)
N[0,1] = N[1,0] = N[1,2] = N[2,1] = 1

# Bayesian optimization object
BO = bayesian_optimization( objective = obj_fun,
                            domain = domain,
                            arg_max = arg_max,
                            n_workers = 3,
                            network = N,
                            kernel = kernels.RBF(length_scale_bounds=(1, 1000.0)),
                            acquisition_function = 'ei',
                            policy = 'greedy',
                            fantasies = 0,
                            regularization = None,
                            regularization_strength = 0.01,
                            grid_density = 30)

# Optimize
BO.optimize(n_iters = 50, n_runs = 1, n_pre_samples = 5, random_search = 1000, plot = False)
for a in range(BO.n_workers):
    print("Predicted max {}: {}".format(a, BO.pre_max[a]))
