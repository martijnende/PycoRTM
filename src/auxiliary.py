import numpy as np


class helper:

    def __init__(self):
        pass

    def compute_mean(self, x):
        N_r, N_theta = self.N_r, self.N_theta
        n = np.arange(N_r * N_theta)
        j = n % N_theta
        i = (n - j) / N_theta + 0.5
        dr = self.dr
        dtheta = self.dtheta

        A = i * dtheta * dr**2
        A_tot = 0.5 * self.theta_range * self.R**2

        x_mean = ((x*A) / A_tot).sum()

        return x_mean

