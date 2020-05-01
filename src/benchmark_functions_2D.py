import numpy as np

class Bohachevsky_1:
    def __init__(self):
        self.domain = np.array([[-100, 100], [-100, 100]])
        self.function = lambda x: x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1])+0.7
        self.min = 0
        self.arg_min = np.array([[0, 0]])

class Rosenbrock:
    def __init__(self):
        self.domain = np.array([[-2, 2], [-1, 3]])
        self.function = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
        self.min = 0
        self.arg_min = np.array([[1, 1]])

class Ackley:
    def __init__(self):
        self.domain = np.array([[-5, 5], [-5, 5]])
        self.function = lambda x: -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + np.exp(1) + 20
        self.min = 0
        self.arg_min = np.array([[0,0]])


class Rastrigin:
    def __init__(self):
        self.domain = np.array([[-5.12, 5.12], [-5.12, 5.12]])
        self.function = lambda x: 20 + x[0]**2 - 10*np.cos(2*np.pi*x[0]) + x[1]**2 - 10*np.cos(2*np.pi*x[1])
        self.min = 0
        self.arg_min = np.array([[0,0]])


class Himmelblau:
    def __init__(self):
        self.domain = np.array([[-5, 5], [-5, 5]])
        self.function = lambda x: (x[0]**(2)+x[1]-11)**(2)+(x[0]+x[1]**(2)-7)**(2)
        self.min = 0
        self.arg_min = np.array([[3,2],[-2.805118,3.131312],[-3.779310,-3.283186],[3.584428,-1.848126]])


class Eggholder:
    def __init__(self):
        self.domain = np.array([[-512, 512], [-512, 512]])
        self.function = lambda x: -(x[1]+47)*np.sin(np.sqrt(np.abs((x[0]/2) + (x[1]+47)))) - x[0]*np.sin(np.sqrt(np.abs(x[0] - (x[1]+47))))
        self.min = -959.6407
        self.arg_min = np.array([[512,404.2319]])


class GoldsteinPrice:
    def __init__(self):
        self.domain = np.array([[-2, 2], [-2, 2]])
        self.function = lambda x: (1+(x[0]+x[1]+1)**2 * (19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2))*(30 + (2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2))
        self.min = 3
        self.arg_min = np.array([[0,-1]])
