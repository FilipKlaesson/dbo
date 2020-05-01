import numpy as np

class Bohachevsky_1:
    def __init__(self):
        self.domain = np.array([[-100, 100], [-100, 100]])
        self.function = lambda x: x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1])+0.7
        self.min = 0
        self.arg_min = np.array([0, 0])

class Rosenbrock:
    def __init__(self):
        self.domain = np.array([[-5, 10], [-5, 10]])
        self.function = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
        self.min = 0
        self.arg_min = np.array([1, 1])

class Ackley_2:
    def __init__(self):
        self.domain = np.array([[-32, 32], [-32, 32]])
        self.function = lambda x: -200*np.exp(-0.2*np.sqrt(x[0]**2 + x[1]**2))
        self.min = -200
        self.arg_min = np.array([0, 0])

class Ackley_3:
    def __init__(self):
        self.domain = np.array([[-32, 32], [-32, 32]])
        self.function = lambda x: -200*np.exp(-0.2*np.sqrt(x[0]**2 + x[1]**2)) + 5*np.exp(np.cos(3*x[0]) + np.sin(3*x[1]))
        self.min = −195.629028238419
        self.arg_min = np.array([0.682584587365898, -0.36075325513719])
