import os
import csv
import copy
import imageio
import datetime
import warnings
import itertools
import __main__
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tqdm import tqdm
from matplotlib import cm
from scipy.stats import norm
from scipy.optimize import minimize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import kernels, GaussianProcessRegressor

warnings.filterwarnings("ignore")

class bayesian_optimization:
    def __init__(self, objective, domain, arg_max = None, n_workers = 1, network = None, kernel = kernels.RBF(), alpha=10**(-10), acquisition_function = 'ei', policy = 'greedy', fantasies = 0, epsilon = 0.01, regularization = None, regularization_strength = None, pending_regularization = None, pending_regularization_strength = None, grid_density = 100):

        # Optimization setup
        self.objective = objective
        self.n_workers = n_workers
        if network is None:
            self.network = np.eye(n_workers)
        else:
            self.network = network
        self._policy = policy
        if policy not in ['greedy', 'boltzmann']:
            print("Supported policies: 'greedy', 'boltzmann' ")
            return

        # Acquisition function
        if acquisition_function == 'ei':
            self._acquisition_function = self._expected_improvement
        elif acquisition_function == 'ts':
            self._acquisition_function = self._thompson_sampling
        else:
            print('Supported acquisition functions: ei, ts')
            return
        self._epsilon = epsilon
        self._num_fantasies = fantasies

        # Regularization function
        self._regularization = None
        if regularization is not None:
            if regularization == 'ridge':
                self._regularization = self._ridge
            else:
                print('Supported regularization functions: ridge')
                return
        self._pending_regularization = None
        if pending_regularization is not None:
            if pending_regularization == 'ridge':
                self._pending_regularization = self._ridge
            else:
                print('Supported pending_regularization functions: ridge')
                return

        # Domain
        self.domain = domain    #shape = [n_params, 2]
        self._dim = domain.shape[0]
        self._grid_density = grid_density
        grid_elemets = []
        for [i,j] in self.domain:
            grid_elemets.append(np.linspace(i, j, self._grid_density))
        self._grid = np.array(list(itertools.product(*grid_elemets)))

        # Global Maximum
        self.arg_max = arg_max
        if self.arg_max is None:
            obj_grid = [self.objective(i) for i in self._grid]
            self.arg_max = np.array(self._grid[np.array(obj_grid).argmax(), :]).reshape(-1, self._dim)

        # Model Setup
        self.alpha = alpha
        self.kernel = kernel
        self._regularization_strength = regularization_strength
        self._pending_regularization_strength = pending_regularization_strength
        self.model = [GaussianProcessRegressor(  kernel=self.kernel,
                                                    alpha=self.alpha,
                                                    n_restarts_optimizer=10)
                                                    for i in range(self.n_workers) ]
        self.scaler = [StandardScaler() for i in range(n_workers)]

        # Data holders
        self.bc_data = None
        self.X_train = self.Y_train = None
        self.X = self.Y = None
        self._acquisition_evaluations = [[] for i in range(n_workers)]

        # Directory setup
        self._DT_ = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self._ROOT_DIR_ = os.path.dirname(os.path.dirname( __main__.__file__ ))
        self._TEMP_DIR_ = os.path.join(self._ROOT_DIR_, "temp")
        self._ID_DIR_ = os.path.join(self._TEMP_DIR_, self._DT_)
        self._DATA_DIR_ = os.path.join(self._ID_DIR_, "data")
        self._FIG_DIR_ = os.path.join(self._ID_DIR_, "fig")
        self._PNG_DIR_ = os.path.join(self._FIG_DIR_, "png")
        self._PDF_DIR_ = os.path.join(self._FIG_DIR_, "pdf")
        self._GIF_DIR_ = os.path.join(self._FIG_DIR_, "gif")
        for path in [self._TEMP_DIR_, self._DATA_DIR_, self._FIG_DIR_, self._PNG_DIR_, self._PDF_DIR_, self._GIF_DIR_]:
            try:
                os.makedirs(path)
            except FileExistsError:
                pass

    def _regret(self, y):
        return self.objective(self.arg_max[0]) - y

    def _mean_regret(self):
        r_mean = [np.mean(self._simple_regret[:,iter]) for iter in range(self._simple_regret.shape[1])]
        r_std = [np.std(self._simple_regret[:,iter]) for iter in range(self._simple_regret.shape[1])]
        # 95% confidence interval
        conf95 = [1.96*rst/self._simple_regret.shape[0] for rst in r_std]
        return range(self._simple_regret.shape[1]), r_mean, conf95

    def _mean_distance_traveled(self):
        d_mean = [np.mean(self._distance_traveled[:,iter]) for iter in range(self._distance_traveled.shape[1])]
        d_std = [np.std(self._distance_traveled[:,iter]) for iter in range(self._distance_traveled.shape[1])]
        # 95% confidence interval
        conf95 = [1.96*dst/self._distance_traveled.shape[0] for dst in d_std]
        return range(self._distance_traveled.shape[1]), d_mean, conf95

    def _save_data(self, data, name):
        with open(self._DATA_DIR_ + '/' + name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for i in zip(*data):
                writer.writerow(i)
        return

    def _ridge(self, x, center = 0):
        return np.linalg.norm(x - center)

    def _regularize(self, x, a, mu, Y_max):

        if self._regularization is None and self._pending_regularization is None:
            mu = mu - self._epsilon
        else:
            # Distance regularization
            if self._regularization is not None:
                if self._regularization == self._ridge:
                    if self._regularization_strength is not None:
                        reg = np.array([self._regularization_strength*self._ridge(i, self.X[a][-1]) for i in x])
                    else:
                        reg = []
                        kernel = copy.deepcopy(self.model[a].kernel_)
                        param = {"length_scale": 0.7*max([d[1]-d[0] for d in self.domain])}
                        kernel.set_params(**param)
                        for i in x:
                            k = float(kernel(np.array([i]), np.array([self.X[a][-1]])))
                            reg.append(0.1*(1 - k))
                        reg = np.array(reg)
                mu = mu - Y_max*reg

            # Pending query regularization
            if self._pending_regularization is not None:
                # Pending queries
                x_p = []
                for neighbour_agent, neighbour in enumerate(self.network[a]):
                    if neighbour and neighbour_agent < a:
                        x_p.append(self._next_query[neighbour_agent])
                x_p = np.array(x_p).reshape(-1, self._dim)
                if self._pending_regularization == self._ridge:
                    if self._pending_regularization_strength is not None:
                        pending_reg = np.array([self._pending_regularization_strength*sum([1/self._ridge(i, xp) for xp in x_p]) for i in x])
                    else:
                        pending_reg = np.array([sum([0.1*float(self.model[a].kernel_(np.array([i]), np.array([xp]))) for xp in x_p]) for i in x])
                mu = mu - Y_max*pending_reg

        return mu

    def _expected_improvement(self, a, x, model = None):
        """
        Expected improvement acquisition function.
        Arguments:
        ----------
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the expected improvement needs to be computed.
        """

        x = x.reshape(-1, self._dim)

        if model is None:
            model = self.model[a]

        Y_max = np.max(model.y_train_)

        mu, sigma = model.predict(x, return_std=True)
        mu = np.squeeze(mu)
        mu = self._regularize(x, a, mu, Y_max)

        with np.errstate(divide='ignore'):
            Z = (mu - Y_max) / sigma
            expected_improvement = (mu - Y_max) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] = 0
            expected_improvement[expected_improvement < 10**(-100)] = 0

        return -1 * expected_improvement

    def _thompson_sampling(self, a, x, model = None, num_samples = 1):
        """
        Thompson sampling acquisition function.
        Arguments:
        ----------
            agent: integer
                Agent id to find next query for.
            model: sklearn model
                Surrogate model used for acquisition function
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the thompson samples needs to be computed.
        """
        x = x.reshape(-1, self._dim)

        if model is None:
            model = self.model[a]

        Y_max = np.max(model.y_train_)

        yts = model.sample_y(x, n_samples=num_samples, random_state = None)

        if num_samples > 1:
            yts = np.squeeze(yts)
        ts = np.squeeze(np.mean(yts, axis=1))
        ts = self._regularize(x, a, ts, Y_max)

        return -1 * ts

    def _expected_acquisition(self, a, x):

        x = x.reshape(-1, self._dim)

        # Pending queries
        x_p = []
        for neighbour_agent, neighbour in enumerate(self.network[a]):
            if neighbour and neighbour_agent < a:
                x_p.append(self._next_query[neighbour_agent])
        x_p = np.array(x_p).reshape(-1, self._dim)

        if not x_p.shape[0]:
            return self._acquisition_function(a, x)
        else:
            # Sample fantasies
            rng = check_random_state(0)
            mu, cov = self.model[a].predict(x_p, return_cov=True)
            mu = self.scaler[a].inverse_transform(mu)
            cov = self.scaler[a].scale_**2 * cov
            if mu.ndim == 1:
                y_fantasies = rng.multivariate_normal(mu, cov, self._num_fantasies).T
            else:
                y_fantasies = [rng.multivariate_normal(mu[:, i], cov, self._num_fantasies).T[:, np.newaxis] for i in range(mu.shape[1])]
                y_fantasies = np.hstack(y_fantasies)

            # models for fantasies
            fantasy_models = [GaussianProcessRegressor( kernel=self.kernel,
                                                        alpha=self.alpha,
                                                        optimizer=None)
                                                        for i in range(self._num_fantasies) ]

            # acquisition over fantasies
            fantasy_acquisition = np.zeros((x.shape[0], self._num_fantasies))
            for i in range(self._num_fantasies):

                f_X_train = self.X_train[a][:]
                f_y_train = self.Y_train[a][:]

                fantasy_scaler = StandardScaler()
                fantasy_scaler.fit(np.array(f_y_train).reshape(-1, 1))

                # add fantasy data
                for xf,yf in zip(x_p, y_fantasies[:,0,i]):
                    f_X_train = np.append(f_X_train, xf).reshape(-1, self._dim)
                    f_y_train = np.append(f_y_train, yf).reshape(-1, 1)

                # fit fantasy surrogate
                f_y_train = fantasy_scaler.transform(f_y_train)
                fantasy_models[i].fit(f_X_train, f_y_train)

                # calculate acqusition
                acquisition = self._acquisition_function(a,x,fantasy_models[i])
                for j in range(x.shape[0]):
                    fantasy_acquisition[:,i] = acquisition

            # compute expected acquisition
            expected_acquisition = np.zeros(x.shape[0])
            for j in range(x.shape[0]):
                expected_acquisition[j] = np.mean(fantasy_acquisition[j,:])

        return expected_acquisition

    def _blotzmann(self, n, x, acq):
        """
        Softmax distribution on acqusition function points for stochastic query selection
        Arguments:
        ----------
            n: integer
                Iteration number.
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the blotzmann needs to be computed and selected from.
            acq: array-like, shape = [n_samples, 1]
                The acqusition function value for x.
        """
        C = max(abs(max(acq)-acq))
        if C > 10**(-2):
            beta = 3*np.log(n+self._initial_data_size+1)/C
            _blotzmann_prob = lambda e: np.exp(beta*e)
            bm = [_blotzmann_prob(e) for e in acq]
            norm_bm = [float(i)/sum(bm) for i in bm]
            idx = np.random.choice(range(x.shape[0]), p=np.squeeze(norm_bm))
        else:
            idx = np.random.choice(range(x.shape[0]))
        return x[idx]

    def _find_next_query(self, n, a, random_search):
        """
        Proposes the next query.
        Arguments:
        ----------
            n: integer
                Iteration number.
            agent: integer
                Agent id to find next query for.
            model: sklearn model
                Surrogate model used for acqusition function
            random_search: integer.
                Number of random samples used to optimize the acquisition function. Default 1000
        """
        # Candidate set
        x = np.random.uniform(self.domain[:, 0], self.domain[:, 1], size=(random_search, self._dim))

        X = x[:]
        if self._record_step:
            X = np.append(self._grid, x).reshape(-1, self._dim)

        # Calculate acquisition function
        if self._num_fantasies:
            acq = - self._expected_acquisition(a, X)
        else:
            # Vanilla acquisition functions
            acq = - self._acquisition_function(a, X)

        if self._record_step:
            self._acquisition_evaluations[a].append(-1*acq[0:self._grid.shape[0]])
            acq = acq[self._grid.shape[0]:]

        # Apply policy
        if self._policy == 'boltzmann':
            # Boltzmann Policy
            x = self._blotzmann(n, x, acq)
        else:
            #Greedy Policy
            x = x[np.argmax(acq), :]

        return x

    def optimize(self, n_iters, n_runs = 1, x0=None, n_pre_samples=5, random_search=100, plot = False):
        """
        Arguments:
        ----------
            n_iters: integer.
                Number of iterations to run the search algorithm.
            x0: array-like, shape = [n_pre_samples, n_params].
                Array of initial points to sample the loss function for. If None, randomly
                samples from the loss function.
            n_pre_samples: integer.
                If x0 is None, samples `n_pre_samples` initial points.
            random_search: integer.
                Flag that indicates whether to perform random search or L-BFGS-B to optimize the acquisition function.
            plot: bool or integer
                If integer, plot iterations with every plot number iteration. If True, plot every interation.
        """

        self._simple_regret = np.zeros((n_runs, n_iters+1))
        self._distance_traveled = np.zeros((n_runs, n_iters+1))

        for run in tqdm(range(n_runs), position=0, leave = None, disable = not n_runs > 1):

            # Reset model and data before each run
            self._next_query = [[] for i in range(self.n_workers)]
            self.bc_data = [[[] for j in range(self.n_workers)] for i in range(self.n_workers)]
            self.X_train = [[] for i in range(self.n_workers)]
            self.Y_train =[[] for i in range(self.n_workers)]
            self.X = [[] for i in range(self.n_workers)]
            self.Y = [[] for i in range(self.n_workers)]
            self.model = [GaussianProcessRegressor(  kernel=self.kernel,
                                                        alpha=self.alpha,
                                                        n_restarts_optimizer=10)
                                                        for i in range(self.n_workers) ]

            # Initial data
            if x0 is None:
                for params in np.random.uniform(self.domain[:, 0], self.domain[:, 1], (n_pre_samples, self.domain.shape[0])):
                    for a in range(self.n_workers):
                        self.X[a].append(params)
                        self.Y[a].append(self.objective(params))
            else:
                # Change definition of x0 to be specfic for each agent
                for params in x0:
                    for a in range(self.n_workers):
                        self.X[a].append(params)
                        self.Y[a].append(self.objective(params))
            self._initial_data_size = len(self.Y[0])


            for n in tqdm(range(n_iters+1), position = n_runs > 1, leave = None):

                # record step indicator
                self._record_step = False
                if plot and n_runs == 1:
                    if n == n_iters or not n % plot:
                        self._record_step = True



                self._prev_bc_data = copy.deepcopy(self.bc_data)

                for a in range(self.n_workers):

                    # Updata data knowledge
                    if n == 0:
                        X = self.X[a]
                        Y = self.Y[a]
                        self.X_train[a] = self.X[a][:]
                        self.Y_train[a] = self.Y[a][:]
                    else:
                        self.X[a].append(self._next_query[a])
                        self.Y[a].append(self.objective(self._next_query[a]))
                        self.X_train[a].append(self._next_query[a])
                        self.Y_train[a].append(self.objective(self._next_query[a]))

                        X = self.X[a]
                        Y = self.Y[a]
                        for transmitter in range(self.n_workers):
                            for (x,y) in self._prev_bc_data[transmitter][a]:
                                X = np.append(X,x).reshape(-1, self._dim)
                                Y = np.append(Y,y).reshape(-1, 1)
                                self.X_train[a].append(x)
                                self.Y_train[a].append(y)

                    # Standardize
                    Y = self.scaler[a].fit_transform(np.array(Y).reshape(-1, 1))
                    # Fit surrogate
                    self.model[a].fit(X, Y)

                    # Find next query
                    x = self._find_next_query(n, a, random_search)
                    self._next_query[a] = x

                    # In case of a "duplicate", randomly sample next query point.
                    if np.any(np.abs(x - self.model[a].X_train_) <= 10**(-7)):
                        x = np.random.uniform(self.domain[:, 0], self.domain[:, 1], self.domain.shape[0])

                    # Broadcast data to neighbours
                    self._broadcast(a,x,self.objective(x))

                # Calculate regret
                self._simple_regret[run,n] = self._regret(np.max([y_max for y_a in self.Y_train for y_max in y_a]))
                # Calculate distance traveled
                if not n:
                    self._distance_traveled[run,n] = 0
                else:
                    self._distance_traveled[run,n] =  self._distance_traveled[run,n-1] + sum([np.linalg.norm(self.X[a][-2] - self.X[a][-1]) for a in range(self.n_workers)])

                # Plot optimization step
                if self._record_step:
                    self._plot_iteration(n, plot)

        self.pre_arg_max = []
        self.pre_max = []
        for a in range(self.n_workers):
            self.pre_arg_max.append(np.array(self.model[a].y_train_).argmax())
            self.pre_max.append(self.model[a].X_train_[np.array(self.model[a].y_train_).argmax()])

        # Compute and plot regret
        iter, r_mean, r_conf95 = self._mean_regret()
        self._plot_regret(iter, r_mean, r_conf95)

        iter, d_mean, d_conf95 = self._mean_distance_traveled()

        # Save data
        self._save_data(data = [iter, r_mean, r_conf95, d_mean, d_conf95], name = 'data')

        # Generate gif
        if plot and n_runs == 1:
            self._generate_gif(n_iters, plot)

    def _broadcast(self, agent, x, y):
        for neighbour_agent, neighbour in enumerate(self.network[agent]):
            if neighbour and neighbour_agent != agent:
                self.bc_data[agent][neighbour_agent].append((x,y))
        return

    def _plot_iteration(self, iter, plot_iter):
        """
        Plots the surrogate and acquisition function.
        """
        mu = []
        std = []
        for a in range(self.n_workers):
            mu_a, std_a = self.model[a].predict(self._grid, return_std=True)
            mu.append(mu_a)
            std.append(std_a)
            acq = [-1 * self._acquisition_evaluations[a][iter//plot_iter] for a in range(self.n_workers)]

        for a in range(self.n_workers):
            mu[a] = self.scaler[a].inverse_transform(mu[a])
            std[a] = self.scaler[a].scale_ * std[a]

        if self._dim == 1:
            self._plot_1d(iter, mu, std, acq)
        elif self._dim == 2:
            self._plot_2d(iter, mu, acq)
        else:
            print("Can't plot for higher dimensional problems.")

    def _plot_1d(self, iter, mu, std, acq):
        cmap = cm.get_cmap('jet')
        rgba = [cmap(i) for i in np.linspace(0,1,self.n_workers)]
        if self.n_workers == 1:
            rgba = ['k']
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10), sharex=True)

        class ScalarFormatterForceFormat(ticker.ScalarFormatter):
            def _set_format(self):
                self.format = "%1.2f"
        fmt = ScalarFormatterForceFormat()
        fmt.set_powerlimits((0,0))
        fmt.useMathText = True

        #Objective function
        y_obj = [self.objective(i) for i in self._grid]
        ax1.plot(self._grid, y_obj, 'k--', lw=1)
        for a in range(self.n_workers):
            # Surrogate plot
            ax1.plot(self._grid, mu[a], color = rgba[a], lw=1)
            ax1.fill_between(np.squeeze(self._grid), np.squeeze(mu[a]) - 2*std[a], np.squeeze(mu[a]) + 2*std[a], color = rgba[a], alpha=0.1)
            ax1.scatter(self.X[a], self.Y[a], color = rgba[a], s=20, zorder=3)
            ax1.yaxis.set_major_formatter(fmt)
            ax1.set_ylim(bottom = -10, top=14)
            ax1.set_xticks(np.linspace(self._grid[0],self._grid[-1], 5))
            # Acquisition function plot
            ax2.plot(self._grid, acq[a], color = rgba[a], lw=1)
            ax2.axvline(self._next_query[a], color = rgba[a], lw=1)
            ax2.set_xlabel("x", fontsize = 16)
            ax2.yaxis.set_major_formatter(fmt)
            ax2.set_xticks(np.linspace(self._grid[0],self._grid[-1], 5))

        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.tick_params(axis='both', which='minor', labelsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.tick_params(axis='both', which='minor', labelsize=16)
        ax1.yaxis.offsetText.set_fontsize(16)
        ax2.yaxis.offsetText.set_fontsize(16)

        # Legends
        if self.n_workers > 1:
            c = 'k'
        else:
            c = rgba[a]
        legend_elements1 = [Line2D([0], [0], linestyle = '--', color='k', lw=0.8, label='Objective'),
                           Line2D([0], [0], color=c, lw=0.8, label='Surrogate'),
                           Line2D([], [], marker='o', color=c, label='Observations', markerfacecolor=c, markersize=4)]
        leg1 = ax1.legend(handles=legend_elements1, fontsize = 16, loc='upper right', fancybox=True, framealpha=0.2)
        ax1.add_artist(leg1)
        ax1.legend(["Iteration %d" % (iter)], fontsize = 16, loc='upper left', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)


        legend_elements2 = [ Line2D([0], [0], color=c, lw=0.8, label='Acquisition'),
                            Line2D([], [], color=c, marker='|', linestyle='None',
                          markersize=10, markeredgewidth=1, label='Next Query')]
        ax2.legend(handles=legend_elements2, fontsize = 16, loc='upper right', fancybox=True, framealpha=0.5)

        plt.tight_layout()
        plt.savefig(self._PDF_DIR_ + '/bo_iteration_%d.pdf' % (iter), bbox_inches='tight')
        plt.savefig(self._PNG_DIR_ + '/bo_iteration_%d.png' % (iter), bbox_inches='tight')

    def _plot_2d(self, iter, mu, acq):

        cmap = cm.get_cmap('jet')
        rgba = [cmap(i) for i in np.linspace(0,1,self.n_workers)]
        if self.n_workers == 1:
            rgba = ['k']

        class ScalarFormatterForceFormat(ticker.ScalarFormatter):
            def _set_format(self):
                self.format = "%1.2f"
        fmt = ScalarFormatterForceFormat()
        fmt.set_powerlimits((0,0))
        fmt.useMathText = True

        x = np.array(self.X)
        y = np.array(self.Y)

        first_param_grid = np.linspace(self.domain[0,0], self.domain[0,1], self._grid_density)
        second_param_grid = np.linspace(self.domain[1,0], self.domain[1,1], self._grid_density)
        X, Y = np.meshgrid(first_param_grid, second_param_grid, indexing='ij')

        for a in range(self.n_workers):

            fig, ax = plt.subplots(3, 1, figsize=(10,10), sharex=True, sharey=True)
            (ax1, ax2, ax3) = ax
            plt.setp(ax.flat, aspect=1.0, adjustable='box')

            N = 100
            # Objective plot
            Y_obj = [self.objective(i) for i in self._grid]
            clev1 = np.linspace(min(Y_obj), max(Y_obj),N)
            cp1 = ax1.contourf(X, Y, np.array(Y_obj).reshape(X.shape), clev1,  cmap = cm.coolwarm)
            for c in cp1.collections:
                c.set_edgecolor("face")
            cbar1 = plt.colorbar(cp1, ax=ax1, shrink = 0.9, format=fmt, pad = 0.05)
            cbar1.ax.tick_params(labelsize=10)
            cbar1.ax.locator_params(nbins=5)
            ax1.autoscale(False)
            ax1.scatter(x[a][:, 0], x[a][:, 1], zorder=1, color = rgba[a], s = 10)
            ax1.axvline(self._next_query[a][0], color='k', linewidth=1)
            ax1.axhline(self._next_query[a][1], color='k', linewidth=1)
            ax1.set_ylabel("y", fontsize = 10, rotation=0)
            leg1 = ax1.legend(['Objective'], fontsize = 10, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax1.add_artist(leg1)
            ax1.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax1.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax1.set_xticks(np.linspace(first_param_grid[0],first_param_grid[-1], 5))
            ax1.set_yticks(np.linspace(second_param_grid[0],second_param_grid[-1], 5))
            plt.setp(ax1.get_yticklabels()[0], visible=False)
            ax1.tick_params(axis='both', which='both', labelsize=10)
            ax1.scatter(self.arg_max[:,0], self.arg_max[:,1], marker='x', c='gold', s=30)

            if self.n_workers > 1:
                ax1.legend(["Iteration %d" % (iter), "Agent %d" % (a)], fontsize = 10, loc='upper left', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            else:
                ax1.legend(["Iteration %d" % (iter)], fontsize = 10, loc='upper left', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)

            # Surrogate plot
            d = 0
            if mu[a].reshape(X.shape).max() - mu[a].reshape(X.shape).min() == 0:
                d = acq[a].reshape(X.shape).max()*0.1
            clev2 = np.linspace(mu[a].reshape(X.shape).min() - d, mu[a].reshape(X.shape).max() + d,N)
            cp2 = ax2.contourf(X, Y, mu[a].reshape(X.shape), clev2,  cmap = cm.coolwarm)
            for c in cp2.collections:
                c.set_edgecolor("face")
            cbar2 = plt.colorbar(cp2, ax=ax2, shrink = 0.9, format=fmt, pad = 0.05)
            cbar2.ax.tick_params(labelsize=10)
            cbar2.ax.locator_params(nbins=5)
            ax2.autoscale(False)
            ax2.scatter(x[a][:, 0], x[a][:, 1], zorder=1, color = rgba[a], s = 10)
            ax2.axvline(self._next_query[a][0], color='k', linewidth=1)
            ax2.axhline(self._next_query[a][1], color='k', linewidth=1)
            ax2.set_ylabel("y", fontsize = 10, rotation=0)
            ax2.legend(['Surrogate'], fontsize = 10, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax2.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax2.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax2.set_xticks(np.linspace(first_param_grid[0],first_param_grid[-1], 5))
            ax2.set_yticks(np.linspace(second_param_grid[0],second_param_grid[-1], 5))
            plt.setp(ax2.get_yticklabels()[0], visible=False)
            plt.setp(ax2.get_yticklabels()[-1], visible=False)
            ax2.tick_params(axis='both', which='both', labelsize=10)

            # Broadcasted data
            for transmitter in range(self.n_workers):
                x_bc = []
                for (xbc,ybc) in self._prev_bc_data[transmitter][a]:
                    x_bc = np.append(x_bc,xbc).reshape(-1, self._dim)
                x_bc = np.array(x_bc)
                if x_bc.shape[0]>0:
                    ax1.scatter(x_bc[:, 0], x_bc[:, 1], zorder=1, color = rgba[transmitter], s = 10)
                    ax2.scatter(x_bc[:, 0], x_bc[:, 1], zorder=1, color = rgba[transmitter], s = 10)

            # Acquisition function contour plot
            d = 0
            if acq[a].reshape(X.shape).max() - acq[a].reshape(X.shape).min() == 0.0:
                d = acq[a].reshape(X.shape).max()*0.1
                d = 10**(-100)
            clev3 = np.linspace(acq[a].reshape(X.shape).min() - d, acq[a].reshape(X.shape).max() + d,N)
            cp3 = ax3.contourf(X, Y, acq[a].reshape(X.shape), clev3, cmap = cm.coolwarm)
            cbar3 = plt.colorbar(cp3, ax=ax3, shrink = 0.9, format=fmt, pad = 0.05)
            for c in cp3.collections:
                c.set_edgecolor("face")
            cbar3.ax.locator_params(nbins=5)
            cbar3.ax.tick_params(labelsize=10)
            ax3.autoscale(False)
            ax3.axvline(self._next_query[a][0], color='k', linewidth=1)
            ax3.axhline(self._next_query[a][1], color='k', linewidth=1)
            ax3.set_xlabel("x", fontsize = 10)
            ax3.set_ylabel("y", fontsize = 10, rotation=0)
            ax3.legend(['Acquisition'], fontsize = 10, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax3.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax3.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax3.set_xticks(np.linspace(first_param_grid[0],first_param_grid[-1], 5))
            ax3.set_yticks(np.linspace(second_param_grid[0],second_param_grid[-1], 5))
            plt.setp(ax3.get_yticklabels()[-1], visible=False)
            ax3.tick_params(axis='both', which='both', labelsize=10)

            ax1.tick_params(axis='both', which='major', labelsize=10)
            ax1.tick_params(axis='both', which='minor', labelsize=10)
            ax2.tick_params(axis='both', which='major', labelsize=10)
            ax2.tick_params(axis='both', which='minor', labelsize=10)
            ax3.tick_params(axis='both', which='major', labelsize=10)
            ax3.tick_params(axis='both', which='minor', labelsize=10)
            ax1.yaxis.offsetText.set_fontsize(10)
            ax2.yaxis.offsetText.set_fontsize(10)
            ax3.yaxis.offsetText.set_fontsize(10)

            fig.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(self._PDF_DIR_ + '/bo_iteration_%d_agent_%d.pdf' % (iter, a), bbox_inches='tight')
            plt.savefig(self._PNG_DIR_ + '/bo_iteration_%d_agent_%d.png' % (iter, a), bbox_inches='tight')

    def _plot_regret(self, iter, r_mean, conf95, log = False):

        use_log_scale = max(r_mean)/min(r_mean) > 10

        if not use_log_scale:
            # absolut error for linear scale
            lower = [r + err for r, err in zip(r_mean, conf95)]
            upper = [r - err for r, err in zip(r_mean, conf95)]
        else:
            # relative error for log scale
            lower = [10**(np.log10(r) + (0.434*err/r)) for r, err in zip(r_mean, conf95)]
            upper = [10**(np.log10(r) - (0.434*err/r)) for r, err in zip(r_mean, conf95)]

        fig = plt.figure()

        if use_log_scale:
            plt.yscale('log')

        plt.plot(iter, r_mean, '-', linewidth=1)
        plt.fill_between(iter, upper, lower, alpha=0.3)
        plt.xlabel('iterations')
        plt.ylabel('immediate regret')
        plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.3)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        if use_log_scale:
            plt.savefig(self._PDF_DIR_ + '/regret_log.pdf', bbox_inches='tight')
            plt.savefig(self._PNG_DIR_ + '/regret_log.png', bbox_inches='tight')
        else:
            plt.savefig(self._PDF_DIR_ + '/regret.pdf', bbox_inches='tight')
            plt.savefig(self._PNG_DIR_ + '/regret.png', bbox_inches='tight')

    def _generate_gif(self, n_iters, plot):
        if self._dim == 1:
            plots = []
            for i in range(n_iters+1):
                if plot is True or i == n_iters:
                    try:
                        plots.append(imageio.imread(self._PNG_DIR_ + '/bo_iteration_%d.png' % (i)))
                    except: pass
                elif not i % plot:
                    try:
                        plots.append(imageio.imread(self._PNG_DIR_ + '/bo_iteration_%d.png' % (i)))
                    except: pass
            imageio.mimsave(self._GIF_DIR_ + '/bo.gif', plots, duration=1.0)
        else:
            for a in range(self.n_workers):
                plots = []
                for i in range(n_iters+1):
                    if plot is True or i == n_iters:
                        try:
                            plots.append(imageio.imread(self._PNG_DIR_ + '/bo_iteration_%d_agent_%d.png' % (i, a)))
                        except: pass
                    elif not i % plot:
                        try:
                            plots.append(imageio.imread(self._PNG_DIR_ + '/bo_iteration_%d_agent_%d.png' % (i, a)))
                        except: pass
                imageio.mimsave(self._GIF_DIR_ + '/bo_agent_%d.gif' % (a), plots, duration=1.0)
