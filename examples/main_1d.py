import numpy as np
import sklearn.gaussian_process.kernels as kernels
from src.bayesian_optimization import bayesian_optimization

# Set numpy random seed
np.random.seed(0)

# Domain
domain = np.array([[-10, 10]])

# Objective function
obj_fun = lambda x: (x[0]-0.5)*np.sin(x[0])

num_agents = 3
# Communication network
N = np.eye(3)
N[0,1] = N[1,0] = N[1,2] = N[2,1] = 1

# Bayesian optimization object
BO = bayesian_optimization( objective = obj_fun,
                            domain = domain,
                            n_workers = num_agents,
                            network = N,
                            kernel = kernels.RBF(),
                            acquisition_function = 'ei',
                            stochastic_policy = False,
                            regularization = None,
                            regularization_strength = 0.01,
                            grid_density = 100 )

# Optimize
BO.optimize(n_iters = 10, n_runs = 1, n_pre_samples = 3, random_search = 1000, plot = True)
for a in range(BO.n_workers):
    print("Predicted max {}: {}".format(a, BO.pre_max[a]))
