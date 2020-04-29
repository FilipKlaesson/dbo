import os
import csv
import datetime
import numpy as np
import itertools
import imageio
import copy
import sklearn.gaussian_process as gp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from matplotlib import cm
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings("ignore")

class bayesian_optimization:
    def __init__(self, obj, domain, arg_max = None, n_workers = 1, network = None, kernel = gp.kernels.RBF(), noise=10**(-5), acquisition_function = 'ei', stochastic_policy = False, regularization = None, l = 0.01, grid_density = 30):

        # Optimization setup
        self.objective = obj
        self.n_workers = n_workers
        if network is None:
            self.network = np.ones((n_workers,n_workers))
        else:
            self.network = network
        self.domain = domain    #shape = [n_params, 2]
        self.dim = domain.shape[0]
        self.grid = None
        self.grid_density = grid_density
        self.scaler = [StandardScaler() for i in range(n_workers)]

        # Model Setup
        self.kernel = kernel
        self.acquisition_function = None
        self.regularization = None
        self.l = l
        self.stochastic_policy = stochastic_policy
        self.model = [gp.GaussianProcessRegressor(  kernel=self.kernel,
                                                    alpha=noise,
                                                    n_restarts_optimizer=10,
                                                    normalize_y = False)
                                                    for i in range(n_workers)]

        # Data info
        self.next_query = [[] for i in range(n_workers)]
        self.initial_data_size = None
        self.predicted_optimum = []
        self.bc_data = [[[] for j in range(n_workers)] for i in range(n_workers)]
        self.X_train = [[] for i in range(self.n_workers)]
        self.Y_train =[[] for i in range(self.n_workers)]
        self.X = [[] for i in range(self.n_workers)]
        self.Y = [[] for i in range(self.n_workers)]
        self.simple_regret = []

        # Directory setup
        self._DT_ = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self._ROOT_DIR_ = os.path.dirname(os.path.dirname( __file__ ))
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

        # Data for plots
        self.thompson_samples = [[] for i in range(n_workers)]

        # Set acquisition function
        if acquisition_function == 'ei':
            self.acquisition_function = self.expected_improvement
        elif acquisition_function == 'ts':
            self.acquisition_function = self.thompson_sampling
        else:
            print('Supported acquisition functions: ei, ts')
            return

        # Set regularization function
        if regularization is not None:
            if regularization == 'ridge':
                self.regularization = self.ridge
            else:
                print('Supported regularization functions: ridge')
                return

        # Build grid
        grid_elemets = []
        for [i,j] in self.domain:
            grid_elemets.append(np.linspace(i, j, self.grid_density))
        self.grid = np.array(list(itertools.product(*grid_elemets)))

        # Find global optimum
        self.arg_max = arg_max
        if self.arg_max is None:
            obj_grid = [self.objective(i) for i in self.grid]
            self.arg_max = self.grid[np.array(obj_grid).argmax(), :]
        self.maximum = self.objective(self.arg_max)

    def regret(self, y):
        self.simple_regret.append(self.maximum - y)

    def save_data(self):
        with open(self._DATA_DIR_ + '/regret.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i in self.simple_regret:
                writer.writerow([i])
        return

    def ridge(self, x, center = 0):
        return self.l * np.linalg.norm(x - center)

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
                Expected improvment margin, increases exploration
        """

        x = x.reshape(-1, self.dim)

        Y_opt = np.max(self.model[a].y_train_)

        mu, sigma = model.predict(x, return_std=True)
        mu = np.squeeze(mu)

        if self.regularization is None:
            mu = mu - epsilon
        else:
            if self.regularization == self.ridge:
                ridge = np.array([self.ridge(i, self.X[a][-1]) for i in x])
                mu = mu - Y_opt*ridge

        with np.errstate(divide='ignore'):
            Z = (mu - Y_opt) / sigma
            expected_improvement = (mu - Y_opt) * norm.cdf(Z) + sigma * norm.pdf(Z)
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
        x = x.reshape(-1, self.dim)
        x_grid = np.append(self.grid, x).reshape(-1, self.dim)

        ns = 1
        yts = model.sample_y(x_grid, n_samples=ns)
        if ns > 1:
            yts = np.squeeze(yts)
        ts = np.mean(yts, axis=1)

        ts_grid = ts[0:self.grid.shape[0]]
        ts_x = ts[self.grid.shape[0]:]

        self.thompson_samples[a].append(-1*ts_grid)

        return -1 * ts_x

    def softmax(self, n, x, acq):
        """
        Softmax distribution on acqusition function points for stochastic query selection
        Arguments:
        ----------
            n: integer
                Iteration number.
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the softmax needs to be computed and selected from.
            acq: array-like, shape = [n_samples, 1]
                The acqusition function value for x.
        """
        C = max(abs(max(acq)-acq))
        if C > 0:
            beta = 2*np.log(n+self.initial_data_size+1)/C
            softmax_prob = lambda e: np.exp(beta*e)
            sm = [softmax_prob(e) for e in acq]
            norm_sm = [float(i)/sum(sm) for i in sm]
            idx = np.random.choice(range(x.shape[0]), p=np.squeeze(norm_sm))
        else:
            idx = np.random.choice(range(x.shape[0]))
        return x[idx]

    def find_next_query(self, n, a, model, random_search):
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
        x = np.random.uniform(self.domain[:, 0], self.domain[:, 1], size=(random_search, self.dim))
        if self.acquisition_function == self.expected_improvement:
            ei = - self.expected_improvement(model, x, a)
            #Stochastic Boltzmann Policy
            if self.stochastic_policy:
                x = self.softmax(n, x, ei)
            #Greedy Policy
            else:
                x = x[np.argmax(ei), :]
        else:
            ts = - self.thompson_sampling(a, model, x)
            x = x[np.argmax(ts), :]
        return x

    def optimize(self, n_iters, n_runs = 1, x0=None, n_pre_samples=5, random_search=100, alpha=1e-5, epsilon=1e-7, plot = False):
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
            alpha: double.
                Variance of the error term of the GP.
            epsilon: double.
                Precision tolerance for floats.
            plot: bool or integer
                If integer, plot iterations with every plot number iteration. If True, plot every interation.
        """

        # Initial data
        for a in range(self.n_workers):
            if x0 is None:
                for params in np.random.uniform(self.domain[:, 0], self.domain[:, 1], (n_pre_samples, self.domain.shape[0])):
                    self.X[a].append(params)
                    self.Y[a].append(self.objective(params))
            else:
                for params in x0:
                    self.X[a].append(params)
                    self.Y[a].append(self.objective(params))
            self.initial_data_size = len(self.Y[0])

        for n in tqdm(range(n_iters+1), position=0):

            self.prev_bc_data = copy.deepcopy(self.bc_data)

            for a in range(self.n_workers):

                # Updata data knowledge
                if n == 0:
                    X = self.X[a]
                    Y = self.Y[a]
                    self.X_train[a] = self.X[a][:]
                    self.Y_train[a] = self.Y[a][:]
                else:
                    self.X[a].append(self.next_query[a])
                    self.Y[a].append(self.objective(self.next_query[a]))
                    self.X_train[a].append(self.next_query[a])
                    self.Y_train[a].append(self.objective(self.next_query[a]))

                    X = self.X[a]
                    Y = self.Y[a]
                    for transmitter in range(self.n_workers):
                        for (x,y) in self.prev_bc_data[transmitter][a]:
                            X = np.append(X,x).reshape(-1, self.dim)
                            Y = np.append(Y,y).reshape(-1, 1)
                            self.X_train[a].append(x)
                            self.Y_train[a].append(y)

                # Standardize
                Y = self.scaler[a].fit_transform(np.array(Y).reshape(-1, 1))
                # Fit surrogate
                self.model[a].fit(X, Y)

                # Find next query
                x = self.find_next_query(n, a, self.model[a], random_search)
                self.next_query[a] = x

                # Non-positive definite covariance matrix break the Gaussian process.
                # In case of a "duplicate", randomly sample a next query point.
                if np.any(np.abs(x - self.model[a].X_train_) <= epsilon):
                    x = np.random.uniform(self.domain[:, 0], self.domain[:, 1], self.domain.shape[0])

                # Broadcast data to neighbours
                self.broadcast(a,x,self.objective(x))

            self.regret(np.max([y_max for y_a in self.Y_train for y_max in y_a]))

            if plot is not False:
                if plot is True or n == n_iters:
                    self.plot_iteration(n)
                elif not n % plot:
                    self.plot_iteration(n)

        for a in range(self.n_workers):
            self.predicted_optimum.append(self.model[a].X_train_[np.array(self.model[a].y_train_).argmax()])

        if plot is not False:
            self.generate_gif(n_iters, plot)

        self.save_data()

    def broadcast(self, agent, x, y):
        for neighbour_agent, neighbour in enumerate(self.network[agent]):
            if neighbour and neighbour_agent != agent:
                self.bc_data[agent][neighbour_agent].append((x,y))
        return

    def plot_iteration(self, iter):
        """
        Plots the surrogate and acquisition function.
        """
        mu = []
        std = []
        for a in range(self.n_workers):
            mu_a, std_a = self.model[a].predict(self.grid, return_std=True)
            mu.append(mu_a)
            std.append(std_a)
        if self.acquisition_function == self.expected_improvement:
            acq = [-1 * self.expected_improvement(self.model[a], self.grid, a) for a in range(self.n_workers)]
        else:
            acq = [-1 * self.thompson_samples[a][n] for a in range(self.n_workers)]

        for a in range(self.n_workers):
            mu[a] = self.scaler[a].inverse_transform(mu[a])
            std[a] = self.scaler[a].scale_ * std[a]

        if self.dim == 1:
            self._plot_1d(iter, mu, std, acq)
        elif self.dim == 2:
            self._plot_2d(iter, mu, acq)
        else:
            print("Can't plot for higher dimensional problems.")

    def _plot_1d(self, iter, mu, std, acq):
        cmap = cm.get_cmap('jet')
        rgba = [cmap(i) for i in np.linspace(0,1,self.n_workers)]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8), sharex=True)

        #Objective function
        y_obj = [self.objective(i) for i in self.grid]
        ax1.plot(self.grid, y_obj, 'k--')
        for a in range(self.n_workers):
            # Surrogate plot
            ax1.plot(self.grid, mu[a], color = rgba[a])
            ax1.fill_between(np.squeeze(self.grid), np.squeeze(mu[a]) - std[a], np.squeeze(mu[a]) + std[a], color = rgba[a], alpha=0.2)
            ax1.scatter(self.X[a], self.Y[a], color = rgba[a], s=20, zorder=3)
            # Acquisition function plot
            ax2.plot(self.grid, acq[a], color = rgba[a])
            ax2.axvline(self.next_query[a], color = rgba[a])
            ax2.set_xlabel("x")

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

        first_param_grid = np.linspace(self.domain[0,0], self.domain[0,1], self.grid_density)
        second_param_grid = np.linspace(self.domain[1,0], self.domain[1,1], self.grid_density)
        X, Y = np.meshgrid(first_param_grid, second_param_grid, indexing='ij')

        for a in range(self.n_workers):

            fig, ax = plt.subplots(3, 1, figsize=(10,10), sharex=True, sharey=True)
            (ax1, ax2, ax3) = ax
            plt.setp(ax.flat, aspect=1.0, adjustable='box')

            # Objective plot
            Y_obj = [self.objective(i) for i in self.grid]
            clev1 = np.linspace(min(Y_obj), max(Y_obj),100)
            cp1 = ax1.contourf(X, Y, np.array(Y_obj).reshape(X.shape), clev1,  cmap = cm.coolwarm)
            cbar1 = plt.colorbar(cp1, ax=ax1, shrink = 0.9, format=fmt)
            cbar1.ax.tick_params(labelsize=7)
            ax1.autoscale(False)
            ax1.scatter(x[a][:, 0], x[a][:, 1], zorder=1, color = rgba[a], s = 10)
            ax1.axvline(self.next_query[a][0], color='k', linewidth=1)
            ax1.axhline(self.next_query[a][1], color='k', linewidth=1)
            ax1.set_title("Iteration %d" % (iter))
            ax1.set_ylabel("y")
            ax1.legend(['Objective'], fontsize = 8, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax1.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax1.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax1.set_xticks(np.linspace(first_param_grid[0]+1,first_param_grid[-1]-1, 5))
            ax1.set_yticks(np.linspace(second_param_grid[0]+1,second_param_grid[-1]-1, 5))
            ax1.tick_params(axis='both', which='both', labelsize=7)
            ax1.scatter(self.arg_max[0], self.arg_max[1], marker='x', c='gold', s=50)

            # Surrogate plot
            d = 0
            if mu[a].reshape(X.shape).max() - mu[a].reshape(X.shape).min() == 0:
                d = acq[a].reshape(X.shape).max()*0.1
            clev2 = np.linspace(mu[a].reshape(X.shape).min() - d, mu[a].reshape(X.shape).max() + d,100)
            cp2 = ax2.contourf(X, Y, mu[a].reshape(X.shape), clev2,  cmap = cm.coolwarm)
            cbar2 = plt.colorbar(cp2, ax=ax2, shrink = 0.9, format=fmt)
            cbar2.ax.tick_params(labelsize=7)
            ax2.autoscale(False)
            ax2.scatter(x[a][:, 0], x[a][:, 1], zorder=1, color = rgba[a], s = 10)
            ax2.axvline(self.next_query[a][0], color='k', linewidth=1)
            ax2.axhline(self.next_query[a][1], color='k', linewidth=1)
            ax2.set_ylabel("y")
            ax2.legend(['Surrogate'], fontsize = 8, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax2.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax2.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax2.set_xticks(np.linspace(first_param_grid[0]+1,first_param_grid[-1]-1, 5))
            ax2.set_yticks(np.linspace(second_param_grid[0]+1,second_param_grid[-1]-1, 5))
            ax2.tick_params(axis='both', which='both', labelsize=7)

            # Broadcasted data
            for transmitter in range(self.n_workers):
                x_bc = []
                for (xbc,ybc) in self.prev_bc_data[transmitter][a]:
                    x_bc = np.append(x_bc,xbc).reshape(-1, self.dim)
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
            cbar3 = plt.colorbar(cp3, ax=ax3, shrink = 0.9, format=fmt)
            cbar3.ax.tick_params(labelsize=7)
            ax3.autoscale(False)
            ax3.axvline(self.next_query[a][0], color='k', linewidth=1)
            ax3.axhline(self.next_query[a][1], color='k', linewidth=1)
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
            ax3.legend(['Acquisition'], fontsize = 8, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax3.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax3.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax3.set_xticks(np.linspace(first_param_grid[0]+1,first_param_grid[-1]-1, 5))
            ax3.set_yticks(np.linspace(second_param_grid[0]+1,second_param_grid[-1]-1, 5))
            ax3.tick_params(axis='both', which='both', labelsize=7)

            fig.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(self._PDF_DIR_ + '/bo_iteration_%d_agent_%d.pdf' % (iter, a), bbox_inches='tight')
            plt.savefig(self._PNG_DIR_ + '/bo_iteration_%d_agent_%d.png' % (iter, a), bbox_inches='tight')

    def generate_gif(self, n_iters, plot):
        if self.dim == 1:
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
