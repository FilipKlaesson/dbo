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
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import kernels, GaussianProcessRegressor

warnings.filterwarnings("ignore")

class bayesian_optimization:
    def __init__(self, objective, domain, arg_max = None, n_workers = 1, network = None, kernel = kernels.RBF(), alpha=10**(-10), acquisition_function = 'ei', stochastic_policy = False, regularization = None, regularization_strength = 0.01, grid_density = 100):

        # Optimization setup
        self.objective = objective
        self.n_workers = n_workers
        if network is None:
            self.network = np.eye(n_workers)
        else:
            self.network = network

        # Acquisition function
        if acquisition_function == 'ei':
            self._acquisition_function = self.expected_improvement
        elif acquisition_function == 'ts':
            self._acquisition_function = self.thompson_sampling
        else:
            print('Supported acquisition functions: ei, ts')
            return

        # Regularization function
        self._regularization = None
        if regularization is not None:
            if regularization == 'ridge':
                self._regularization = self.ridge
            else:
                print('Supported regularization functions: ridge')
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
        self._stochastic_policy = stochastic_policy
        self.model = [GaussianProcessRegressor(  kernel=self.kernel,
                                                    alpha=self.alpha,
                                                    n_restarts_optimizer=10)
                                                    for i in range(self.n_workers) ]
        self.scaler = [StandardScaler() for i in range(n_workers)]

        # Data holders
        self.bc_data = None
        self.X_train = self.Y_train = None
        self.X = self.Y = None
        self._thompson_samples = [[] for i in range(n_workers)]

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
        return r_mean, conf95

    def _save_data(self, data, name):
        with open(self._DATA_DIR_ + '/' + name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for i in zip(*data):
                writer.writerow(i)
        return

    def ridge(self, x, center = 0):
        return self._regularization_strength * np.linalg.norm(x - center)

    def expected_improvement(self, model, x, a, epsilon = 0.01):
        """
        Expected improvement acquisition function.
        Arguments:
        ----------
            model: sklearn model
                Surrogate model used for acquisition function
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the expected improvement needs to be computed.
            epsilon: float
                Expected improvement margin, increases exploration
        """

        x = x.reshape(-1, self._dim)

        Y_max = np.max(self.model[a].y_train_)

        mu, sigma = model.predict(x, return_std=True)
        mu = np.squeeze(mu)

        if self._regularization is None:
            mu = mu - epsilon
        else:
            if self._regularization == self.ridge:
                ridge = np.array([self.ridge(i, self.X[a][-1]) for i in x])
                mu = mu - Y_max*ridge

        with np.errstate(divide='ignore'):
            Z = (mu - Y_max) / sigma
            expected_improvement = (mu - Y_max) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] = 0
            expected_improvement[expected_improvement < 10**(-100)] = 0

        return -1 * expected_improvement

    def thompson_sampling(self, a, model, x):
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
        x_grid = np.append(self._grid, x).reshape(-1, self._dim)

        ns = 1
        yts = model.sample_y(x_grid, n_samples=ns)
        if ns > 1:
            yts = np.squeeze(yts)
        ts = np.mean(yts, axis=1)

        ts_grid = ts[0:self._grid.shape[0]]
        ts_x = ts[self._grid.shape[0]:]

        self._thompson_samples[a].append(-1*ts_grid)

        return -1 * ts_x

    def _softmax(self, n, x, acq):
        """
        Softmax distribution on acqusition function points for stochastic query selection
        Arguments:
        ----------
            n: integer
                Iteration number.
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the _softmax needs to be computed and selected from.
            acq: array-like, shape = [n_samples, 1]
                The acqusition function value for x.
        """
        C = max(abs(max(acq)-acq))
        if C > 0:
            beta = 2*np.log(n+self._initial_data_size+1)/C
            _softmax_prob = lambda e: np.exp(beta*e)
            sm = [_softmax_prob(e) for e in acq]
            norm_sm = [float(i)/sum(sm) for i in sm]
            idx = np.random.choice(range(x.shape[0]), p=np.squeeze(norm_sm))
        else:
            idx = np.random.choice(range(x.shape[0]))
        return x[idx]

    def _next_query(self, n, a, model, random_search):
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
        x = np.random.uniform(self.domain[:, 0], self.domain[:, 1], size=(random_search, self._dim))
        if self._acquisition_function == self.expected_improvement:
            ei = - self.expected_improvement(model, x, a)
            #Stochastic Boltzmann Policy
            if self._stochastic_policy:
                x = self._softmax(n, x, ei)
            #Greedy Policy
            else:
                x = x[np.argmax(ei), :]
        else:
            ts = - self.thompson_sampling(a, model, x)
            x = x[np.argmax(ts), :]
        return x

    def optimize(self, n_iters, n_runs = 1, x0=None, n_pre_samples=5, random_search=100, epsilon=1e-7, plot = False):
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
            epsilon: double.
                Precision tolerance for floats.
            plot: bool or integer
                If integer, plot iterations with every plot number iteration. If True, plot every interation.
        """

        self._simple_regret = np.zeros((n_runs, n_iters+1))

        for run in tqdm(range(n_runs), position=0, leave = None, disable = not n_runs > 1):

            # Reset model and data before each run
            self.__next_query = [[] for i in range(self.n_workers)]
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
            for a in range(self.n_workers):
                if x0 is None:
                    for params in np.random.uniform(self.domain[:, 0], self.domain[:, 1], (n_pre_samples, self.domain.shape[0])):
                        self.X[a].append(params)
                        self.Y[a].append(self.objective(params))
                else:
                    # Change definition of x0 to be specfic for each agent
                    for params in x0:
                        self.X[a].append(params)
                        self.Y[a].append(self.objective(params))
                self._initial_data_size = len(self.Y[0])


            for n in tqdm(range(n_iters+1), position = n_runs > 1, leave = None):

                self._prev_bc_data = copy.deepcopy(self.bc_data)

                for a in range(self.n_workers):

                    # Updata data knowledge
                    if n == 0:
                        X = self.X[a]
                        Y = self.Y[a]
                        self.X_train[a] = self.X[a][:]
                        self.Y_train[a] = self.Y[a][:]
                    else:
                        self.X[a].append(self.__next_query[a])
                        self.Y[a].append(self.objective(self.__next_query[a]))
                        self.X_train[a].append(self.__next_query[a])
                        self.Y_train[a].append(self.objective(self.__next_query[a]))

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
                    x = self._next_query(n, a, self.model[a], random_search)
                    self.__next_query[a] = x

                    # In case of a "duplicate", randomly sample a next query point.
                    if np.any(np.abs(x - self.model[a].X_train_) <= epsilon):
                        x = np.random.uniform(self.domain[:, 0], self.domain[:, 1], self.domain.shape[0])

                    # Broadcast data to neighbours
                    self._broadcast(a,x,self.objective(x))

                # Calculate regret
                self._simple_regret[run,n] = self._regret(np.max([y_max for y_a in self.Y_train for y_max in y_a]))

                # Plot optimization step
                if n_runs == 1 and plot is not False:
                    if plot is True or n == n_iters:
                        self._plot_iteration(n)
                    elif not n % plot:
                        self._plot_iteration(n)

        self.pre_arg_max = []
        self.pre_max = []
        for a in range(self.n_workers):
            self.pre_arg_max.append(np.array(self.model[a].y_train_).argmax())
            self.pre_max.append(self.model[a].X_train_[np.array(self.model[a].y_train_).argmax()])

        # Compute and plot regret
        r_mean, conf95 = self._mean_regret()
        self._plot_regret(r_mean, conf95)

        # Save data
        self._save_data(data = [r_mean, conf95], name = 'regret')

        # Generate gif
        if n_runs == 1:
            if plot is not False:
                self._generate_gif(n_iters, plot)

    def _broadcast(self, agent, x, y):
        for neighbour_agent, neighbour in enumerate(self.network[agent]):
            if neighbour and neighbour_agent != agent:
                self.bc_data[agent][neighbour_agent].append((x,y))
        return

    def _plot_iteration(self, iter):
        """
        Plots the surrogate and acquisition function.
        """
        mu = []
        std = []
        for a in range(self.n_workers):
            mu_a, std_a = self.model[a].predict(self._grid, return_std=True)
            mu.append(mu_a)
            std.append(std_a)
        if self._acquisition_function == self.expected_improvement:
            acq = [-1 * self.expected_improvement(self.model[a], self._grid, a) for a in range(self.n_workers)]
        else:
            acq = [-1 * self._thompson_samples[a][iter] for a in range(self.n_workers)]

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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8), sharex=True)

        class ScalarFormatterForceFormat(ticker.ScalarFormatter):
            def _set_format(self):
                self.format = "%1.2f"
        fmt = ScalarFormatterForceFormat()
        fmt.set_powerlimits((0,0))
        fmt.useMathText = True

        #Objective function
        y_obj = [self.objective(i) for i in self._grid]
        ax1.plot(self._grid, y_obj, 'k--')
        for a in range(self.n_workers):
            # Surrogate plot
            ax1.plot(self._grid, mu[a], color = rgba[a])
            ax1.fill_between(np.squeeze(self._grid), np.squeeze(mu[a]) - std[a], np.squeeze(mu[a]) + std[a], color = rgba[a], alpha=0.2)
            ax1.scatter(self.X[a], self.Y[a], color = rgba[a], s=20, zorder=3)
            ax1.yaxis.set_major_formatter(fmt)
            ax1.set_xticks(np.linspace(self._grid[0],self._grid[-1], 5))
            # Acquisition function plot
            ax2.plot(self._grid, acq[a], color = rgba[a])
            ax2.axvline(self.__next_query[a], color = rgba[a])
            ax2.set_xlabel("x")
            ax2.yaxis.set_major_formatter(fmt)
            ax2.set_xticks(np.linspace(self._grid[0],self._grid[-1], 5))

        # Legends
        if self.n_workers > 1:
            c = 'k'
        else:
            c = rgba[a]
        legend_elements1 = [Line2D([0], [0], linestyle = '--', color='k', lw=1, label='Objective'),
                           Line2D([0], [0], color=c, lw=1, label='Surrogate'),
                           Line2D([], [], marker='o', color=c, label='Queries', markerfacecolor=c, markersize=4)]
        ax1.legend(handles=legend_elements1, loc='upper right', fancybox=True, framealpha=0.5)

        legend_elements2 = [ Line2D([0], [0], color=c, lw=1, label='Acquisition'),
                            Line2D([], [], color=c, marker='|', linestyle='None',
                          markersize=10, markeredgewidth=1, label='Next Query')]
        ax2.legend(handles=legend_elements2, loc='upper right', fancybox=True, framealpha=0.5)

        plt.tight_layout()
        plt.savefig(self._PDF_DIR_ + '/bo_iteration_%d.pdf' % (iter), bbox_inches='tight')
        plt.savefig(self._PNG_DIR_ + '/bo_iteration_%d.png' % (iter), bbox_inches='tight')

    def _plot_2d(self, iter, mu, acq):

        cmap = cm.get_cmap('jet')
        rgba = [cmap(i) for i in np.linspace(0,1,self.n_workers)]

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

            # Objective plot
            Y_obj = [self.objective(i) for i in self._grid]
            clev1 = np.linspace(min(Y_obj), max(Y_obj),100)
            cp1 = ax1.contourf(X, Y, np.array(Y_obj).reshape(X.shape), clev1,  cmap = cm.coolwarm)
            cbar1 = plt.colorbar(cp1, ax=ax1, shrink = 0.9, format=fmt, pad = 0.05)
            cbar1.ax.tick_params(labelsize=7)
            ax1.autoscale(False)
            ax1.scatter(x[a][:, 0], x[a][:, 1], zorder=1, color = rgba[a], s = 10)
            ax1.axvline(self.__next_query[a][0], color='k', linewidth=1)
            ax1.axhline(self.__next_query[a][1], color='k', linewidth=1)
            ax1.set_ylabel("y")
            leg1 = ax1.legend(['Objective'], fontsize = 8, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax1.add_artist(leg1)
            ax1.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax1.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax1.set_xticks(np.linspace(first_param_grid[0],first_param_grid[-1], 5))
            ax1.set_yticks(np.linspace(second_param_grid[0],second_param_grid[-1], 5))
            plt.setp(ax1.get_yticklabels()[0], visible=False)
            ax1.tick_params(axis='both', which='both', labelsize=7)
            ax1.scatter(self.arg_max[:,0], self.arg_max[:,1], marker='x', c='gold', s=50)
            ax1.legend(["Iteration %d" % (iter)], fontsize = 8, loc='upper left', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)

            # Surrogate plot
            d = 0
            if mu[a].reshape(X.shape).max() - mu[a].reshape(X.shape).min() == 0:
                d = acq[a].reshape(X.shape).max()*0.1
            clev2 = np.linspace(mu[a].reshape(X.shape).min() - d, mu[a].reshape(X.shape).max() + d,100)
            cp2 = ax2.contourf(X, Y, mu[a].reshape(X.shape), clev2,  cmap = cm.coolwarm)
            cbar2 = plt.colorbar(cp2, ax=ax2, shrink = 0.9, format=fmt, pad = 0.05)
            cbar2.ax.tick_params(labelsize=7)
            ax2.autoscale(False)
            ax2.scatter(x[a][:, 0], x[a][:, 1], zorder=1, color = rgba[a], s = 10)
            ax2.axvline(self.__next_query[a][0], color='k', linewidth=1)
            ax2.axhline(self.__next_query[a][1], color='k', linewidth=1)
            ax2.set_ylabel("y")
            ax2.legend(['Surrogate'], fontsize = 8, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax2.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax2.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax2.set_xticks(np.linspace(first_param_grid[0],first_param_grid[-1], 5))
            ax2.set_yticks(np.linspace(second_param_grid[0],second_param_grid[-1], 5))
            plt.setp(ax2.get_yticklabels()[0], visible=False)
            plt.setp(ax2.get_yticklabels()[-1], visible=False)
            ax2.tick_params(axis='both', which='both', labelsize=7)

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
            if acq[a].reshape(X.shape).max() - acq[a].reshape(X.shape).min() == 0:
                d = acq[a].reshape(X.shape).max()*0.1
            clev3 = np.linspace(acq[a].reshape(X.shape).min() - d, acq[a].reshape(X.shape).max() + d,100)
            cp3 = ax3.contourf(X, Y, acq[a].reshape(X.shape), clev3, cmap = cm.coolwarm)
            cbar3 = plt.colorbar(cp3, ax=ax3, shrink = 0.9, format=fmt, pad = 0.05)
            cbar3.ax.tick_params(labelsize=7)
            ax3.autoscale(False)
            ax3.axvline(self.__next_query[a][0], color='k', linewidth=1)
            ax3.axhline(self.__next_query[a][1], color='k', linewidth=1)
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
            ax3.legend(['Acquisition'], fontsize = 8, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax3.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax3.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax3.set_xticks(np.linspace(first_param_grid[0],first_param_grid[-1], 5))
            ax3.set_yticks(np.linspace(second_param_grid[0],second_param_grid[-1], 5))
            plt.setp(ax3.get_yticklabels()[-1], visible=False)
            ax3.tick_params(axis='both', which='both', labelsize=7)

            fig.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(self._PDF_DIR_ + '/bo_iteration_%d_agent_%d.pdf' % (iter, a), bbox_inches='tight')
            plt.savefig(self._PNG_DIR_ + '/bo_iteration_%d_agent_%d.png' % (iter, a), bbox_inches='tight')

    def _plot_regret(self, r_mean, conf95, log = False):

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

        plt.plot(range(self._simple_regret.shape[1]), r_mean, '-', linewidth=1)
        plt.fill_between(range(self._simple_regret.shape[1]), upper, lower, alpha=0.3)
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
