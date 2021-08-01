import numpy as np

class FuzzyMetric(object):
    def __init__(self, n, theta, m = None):
        self.n = n
        self.theta = theta
        self.m = m

class M1(FuzzyMetric):
    def _compatibilities(self, residuals):
        size = np.size(residuals)
        compatibilities = np.zeros(size)
        for i, residual in zip(range(size), residuals):
            if residual > (self.n * self.theta):
                continue 
            compatibilities[i] = (1 - (residual / (self.n * self.theta))) ** self.n
        return compatibilities

class M2(FuzzyMetric):
    def _compatibilities(self, residuals):
        size = np.size(residuals)
        compatibilities = np.zeros(size)
        for i, residual in zip(range(size), residuals):
            if residual > (self.theta):
                continue 
            compatibilities[i] = 1 - ((residual ** self.n) / (self.theta ** self.n))
        return compatibilities

class M3(FuzzyMetric):
    def _compatibilities(self, residuals):
        exponents = (residuals ** self.n) / (self.theta ** self.n) 
        compatibilities = np.exp(-exponents)
        return compatibilities

class M4(FuzzyMetric):
    def _compatibilities(self, residuals):
        if self.m == None:
            self.m = 1
        compatibilities = (self.theta ** self.n) / ((self.theta ** self.n) + (self.m * (residuals ** self.n)))
        return compatibilities
