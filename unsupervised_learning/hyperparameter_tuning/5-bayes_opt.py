#!/usr/bin/env python3
"""
Task 5
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    class BayesianOptimization
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        A function that initializes the class BayesianOptimization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min, max = bounds
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
        :return: X_next, EI
            X_next is a numpy.ndarray of shape (1,) representing the next best
            sample point
            EI is a numpy.ndarray of shape (ac_samples,) containing the
            expected improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is False:
            mu_sample_opt = np.amax(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi
        else:
            mu_sample_opt = np.amin(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi

        Z = np.zeros(sigma.shape)
        for i in range(len(sigma)):
            if sigma[i] != 0:
                Z[i] = imp[i] / sigma[i]
            else:
                Z[i] = 0

        ei = np.zeros(sigma.shape)
        for i in range(len(sigma)):
            if sigma[i] > 0:
                ei[i] = imp[i] * norm.cdf(Z[i]) + sigma[i] * norm.pdf(Z[i])
            else:
                ei[i] = 0

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        :param iterations: the maximum number of iterations to perform
        :return: X_opt, Y_opt
            X_opt is a numpy.ndarray of shape (1,) representing the optimal
            point
            Y_opt is a numpy.ndarray of shape (1,) representing the optimal
            function value
        """
        positions = []
        for i in range(iterations):
            next_x, ei = self.acquisition()
            next_y = self.f(next_x)
            pos = np.argmax(ei)
            if pos in positions:
                positions.append(np.argmax(ei))
                break
            self.gp.update(next_x, next_y)
            positions.append(np.argmax(ei))

        if self.minimize is True:
            X_next = np.argmin(self.gp.Y)
        else:
            X_next = np.argmax(self.gp.Y)
        X_op = self.gp.X[X_next]
        Y_op = self.gp.Y[X_next]

        return X_op, Y_op
