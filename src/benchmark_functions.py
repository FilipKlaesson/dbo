import numpy as np

class Bohachevsky_1:
    def __init__(self):
        self.domain = np.array([[-100, 100], [-100, 100]])
        self.function = lambda x: x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1])+0.7

class Rosenbrock:
    def __init__(self):
        self.domain = np.array([[-5, 10], [-5, 10]])
        self.function = lambda x: 100*(x[1]-x[0]^2)**2 + (1-x[0])**2

class Ackley_2:
    def __init__(self):
        self.domain = np.array([[-32, 32], [-32, 32]])
        self.function = lambda x: -200*np.exp(-0.2*np.sqrt(x[0]**2 + x[1]**2))
