# Distributed Bayesian Optimization for Multi-Agent Systems

author: filipkl@kth.se

---

# Table of Contents
1. [Installation](#setup-instructions-for-debian-like-environments)
2. [Docs](#docs)
    1. [Parameters](#parameters)
    2. [Attributes](#attributes)
    3. [Methods](#methods)
    4. [Output](#Output)
3. [Examples](#examples)

---
# Installation (Debian-like environments)

This package is developed for Python 3.7.

1. (Optional) Set up and activate a virtual environment
```bash
virtualenv -p python3 ~/.venvs/dboenv
source ~/.venvs/dboenv/bin/activate
```

2. (Optional) Set up alias for virtual environment
```bash
echo 'alias dboenv="source ~/.venvs/dboenv/bin/activate"' >> ~/.zshrc
source ~/.zshrc  
```

2. Install python dependencies and dbo
```bash
git clone git@github.com:FilipKlaesson/dbo.git && cd dbo
pip install -r requirements.txt
python setup.py install
```

---

# Docs

The Bayesian optimizer is contained in the class bayesian_optimization in src/bayesian_optimization.

```python
class bayesian_optimization(objective, domain, arg_max = None, n_workers = 1,
                            network = None, kernel = kernels.RBF(), alpha=10**(-10),
                            acquisition_function = 'ei', stochastic_policy = False,
                            regularization = None, regularization_strength = 0.01,
                            grid_density = 100)
```

The class implementation utilizes sklearn [GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor) (Algorithm 2.1 of Gaussian Processes for Machine Learning by Rasmussen and Williams) as model on standardized data.

### Parameters:

<pre>
<b>objective</b>: function
Objective function to be maximized. The function take an input with the same
dimension as the domain and return a float.
</pre>

<pre>
<b>domain</b>: numpy.ndarray
The compact domain the objective function is maximized over. A numpy.ndarray of
shape (dim, 2) where dim is the dimension of the function space. Each row specify the
lower and upper bound of the domain along the corresponding dimension.
</pre>

<pre>
<b>arg_max</b>: numpy.ndarray, optional (default: None)
The point that maximizes the objective function. A numpy.ndarray of shape (dim,)
where dim is the dimension of the function space. If None, arg_max will be
approximated by arg_max on the grid defined by the domain and grid_density.
</pre>

<pre>
<b>n_workers</b>: float, optional (default: 1)
Number of agents used in the distributed setting. Corresponds to the number of
models trained in parallel using corresponding agent queries and data broadcasted
from neighbours in the network.
</pre>

<pre>
<b>network</b>: numpy.ndarray, optional (default: None)
Binary communication network, a numpy.ndarray of shape (n_workers,n_workers).
Agent <i>i</i> and <i>j</i> share queries if element (<i>i</i>,<i>j</i>) is non-zero. If None, the
identity matrix is used.
</pre>

<pre>
<b>kernel</b>: kernel object, optional (default: kernels.RBF())
The kernel (sklearn.gaussian_process.kernels object) specifying the covariance
function of the Gaussian Process. The kernel’s hyperparameters are optimized
using the log-marginal likelihood for training data.
</pre>

<pre>
<b>alpha</b>: float, optional (default: 10**(-10))
Noise level in observations. Alpha is added to the diagonal of the kernel.
An increased noise level can prevent potential numerical issues by ensuring that
the covariance matrix is a positive definite matrix.
</pre>

<pre>
<b>acquisition_function</b>: str, optional (default: 'ei')
The acquisition function used to select the next query. Supported acquisition
functions: 'ei' (Expected Improvement), 'ts' (Thompson Sampling).
</pre>

<pre>
<b>stochastic_policy</b> bool, optional (default: False)
Whether to use stochastic policy or greedy policy when selecting next query.
If True, draw the next query from a Boltzmann distribution where the acquisition
function acts as energy measure. If False, next query is argmax of acquisition function.
</pre>

<pre>
<b>regularization</b>: str, optional (default: None)
The regularization function used when selecting next query. The regularization
penalizes the distance from the previous query point. Supported regularization
functions: 'ridge' (Ridge/L2). If None, no regularization will be applied.
</pre>

<pre>
<b>regularization_strength</b>: float, optional (default: 0.01)
Constant multiplied to regularization function, controls the magnitude of the penalty.
</pre>

<pre>
<b>grid_density</b>: int, optional (default: 100)
Number of points in each dimension in the grid.
</pre>

### Attributes:

<pre>
<b>arg_max</b>: numpy.ndarray
The point that maximizes the objective function. If not provided under initialization,
arg_max will be approximated by arg_max on the grid defined by the domain and grid_density.
</pre>

<pre>
<b>model</b>: GaussianProcessRegressor list
List of GaussianProcessRegressor with length n_workers. Contains the models used by
each agent under training.
</pre>

<pre>
<b>pre_arg_max</b>: numpy.ndarray list
List of predicted argmax with length n_workers. Contains the predicted argmax for
each agent.
</pre>

<pre>
<b>pre_max</b>: numpy.ndarray list
List of predicted maximum with length n_workers. Contains the predicted max of
the objective function for each agent.
</pre>

<pre>
<b>X</b>: nested lists
Nested lists of depth 1. Contains lists of queries performed by each agent.
All queries performed by agent <i>i</i> is contained in X[<i>i</i>].
</pre>

<pre>
<b>Y</b>: nested lists
Nested lists of depth 1. Contains lists of function evaluation for corresponding queries.
All function evaluations observed by agent <i>i</i> is contained in Y[<i>i</i>].
</pre>

<pre>
<b>bc_data</b>: nested lists
Nested lists of depth 2. Contains tuples (x,y) of broadcasted data between agents.
All data broadcasted from agent <i>i</i> to agent <i>j</i> is contained in bc_data[<i>i</i>][<i>j</i>].
</pre>

<pre>
<b>X_train</b>: nested lists
Nested lists of depth 1. Contains lists of feature vectors used for training.
All feature vectors used for training by agent <i>i</i> is contained in X_train[<i>i</i>].
</pre>

<pre>
<b>Y_train</b>: nested lists
Nested lists of depth 1. Contains lists of function values used for training.
All function values used for training by agent <i>i</i> is contained in Y_train[<i>i</i>].
</pre>


### Methods

<pre>
<b>__init__</b>(objective, domain, arg_max = None, n_workers = 1,
                network = None, kernel = kernels.RBF(), alpha=10**(-10),
                acquisition_function = 'ei', stochastic_policy = False,
                regularization = None, regularization_strength = 0.01,
                grid_density = 100)
Initialize self.
</pre>

<pre>
<b>optimize</b>(n_iters, n_runs = 1, x0 = None, n_pre_samples = 5,
                random_search = 100, epsilon = 10**(-7), plot = False)

    <b>n_iters</b>: int
        Number of iterations to run the optimization algorithm.
    <b>n_runs</b>: int, optional (default: 1)
        Number of optimization runs.
    <b>x0</b>: numpy.ndarray, optional (default: None)
        Array with shape (n_pre_samples, n_params) containing initial points.
    <b>n_pre_samples</b>: int, optional (default: 5)
        If x0 is None, sample n_pre_samples initial points uniformly random in the domain.
    <b>random_search</b>: int, optional (default: 100)
        Number of samples used in random search to optimize acquisition function.
    <b>epsilon</b>: double, optional (default: 10**(-7))
        Precision tolerance for floats. In case of a "duplicate", randomly sample next query point.
    <b>plot</b>: int
        Plot state every plot number of iteration. If n_runs > 1, plot is disabled.

Runs the optimization algorithm.
Parameter n_runs allow multiple runs of the same optimization setting to simplify analysis.
</pre>


### Output

The output (generated data/plots/gifs) will be stored in the **temp** folder keyed with date and time:

```
dbo
└───examples
└───src  
└───temp
    └───YYYY-MM-DD_HH:MM:SS
        └───data
        └───fig
            └───png
            └───pdf
            └───gif
```

Running <b>optimize()</b> will generate the following output:

* regret

    * File with mean regret over the n_runs together with the 95% confidence bound error (.csv)

    * Plots of mean regret together with 95% confidence bounds (.png/.pdf)

* bo*

    * Plots of every iteration in the optimization algorithm (.png/.pdf)

    * Gif of the progress in the optimization algorithm      (.gif)

Plots and gifs (except regret plot) are disabled when n_runs > 1.

 ---

# Examples

Single-agent 1D example

```python
import numpy as np
import sklearn.gaussian_process.kernels as kernels
from src.bayesian_optimization import bayesian_optimization

# Domain
domain = np.array([[-10, 10]])

# Objective function
obj_fun = lambda x: (x[0]-0.5)*np.sin(x[0])

# Bayesian optimization object
BO = bayesian_optimization( objective = obj_fun,
                            domain = domain,
                            kernel = kernels.RBF(),
                            acquisition_function = 'ei',
                            grid_density = 100)

# Optimize
BO.optimize(n_iters = 10, n_pre_samples = 3, plot = True)
```

<p align="center">
  <img src="https://github.com/FilipKlaesson/dbo/blob/master/examples/fig/example1/bo.gif" width="600" />
</p>

---

Single-agent 2D example

```python
import numpy as np
import sklearn.gaussian_process.kernels as kernels
from src.bayesian_optimization import bayesian_optimization
from src.benchmark_functions_2D import *

# Benchmark Function
fun = Bohachevsky_1()
domain = fun.domain
obj_fun = lambda x: -1*fun.function(x)
arg_max = fun.arg_min

# Bayesian optimization object
BO = bayesian_optimization( objective = obj_fun,
                            domain = domain,
                            arg_max = arg_max,
                            kernel = kernels.RBF(),
                            acquisition_function = 'ei',
                            grid_density = 30)

# Optimize
BO.optimize(n_iters = 10, n_runs = 1, n_pre_samples = 3, plot = True)

```

<p align="center">
  <img src="https://github.com/FilipKlaesson/dbo/blob/master/examples/fig/example2/bo_agent_0.gif" width="600" />
</p>

---

Multi-agent 2D example

```python
import sklearn.gaussian_process.kernels as kernels
from src.bayesian_optimization import bayesian_optimization
from src.benchmark_functions_2D import *

# Benchmark function
fun = Bohachevsky_1()
domain = fun.domain
obj_fun = lambda x: -1*fun.function(x)

# Communication network
num_agents = 3
N = np.eye(3)
N[0,1] = N[1,0] = N[1,2] = N[2,1] = 1

# Bayesian optimization object
BO = bayesian_optimization( obj = obj_fun,
                            domain = domain,
                            n_workers = num_agents,
                            network = N,
                            kernel = kernels.RBF(),
                            acquisition_function = 'ei',
                            grid_density = 30
                          )

# Optimize
BO.optimize(n_iters = 20, n_pre_samples = 3)
for a in range(BO.n_workers):
    print("Predicted max {}: {}".format(a, BO.pre_max[a]))
```

<p align="center">
  <img src="https://github.com/FilipKlaesson/dbo/blob/master/examples/fig/example2/bo_agent_0.gif" width="600" />
</p>
