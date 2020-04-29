import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
from bayesian_optimization import bayesian_optimization

# Set numpy random seed
np.random.seed(0)

# Domain
domain = np.array([[-10, 10]])

# Objective function
obj_fun = lambda x: (x[0]-0.5)*np.sin(x[0])

# Communication network
num_agents = 3
N = np.eye(3)
N[0,1] = N[1,0] = N[1,2] = N[2,1] = 1

# Bayesian optimimzation object
BO = bayesian_optimization( obj = obj_fun, domain = domain,
                            n_workers = num_agents, network = N,
                            kernel = gp.kernels.RBF(length_scale_bounds=(1, 1000)),
                            acquisition_function = 'ei',
                            stochastic_policy = False,
                            regularization = None, l = 0.01,
                            grid_density = 1000 )

# Optimize
BO.optimize(n_iters = 10, n_pre_samples = 1, random_search = 1000, plot = True)
for a in range(BO.n_workers):
    print("Predicted optimum {}: {}".format(a, BO.predicted_optimum[a]))
