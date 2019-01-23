import numpy as np
from math import log10, fabs
from bisect import bisect_right

class rk45_solver:

    # Butcher tableau for RK45 method
    # Time-step coefficients
    t_coeffs = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
    # Other coefficients
    coeffs = np.array([
        [0, 0, 0, 0, 0, 0],
        [1 / 4, 0, 0, 0, 0, 0],
        [3 / 32, 9 / 32, 0, 0, 0, 0],
        [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
        [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
        [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
        [25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0],  # 4th order solution
        [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]  # 5th order solution
    ]).reshape(-1)

    def __init__(self):
        pass

    def rk45_step(self, func, y, t, dt):

        # Prepare solution buffers
        k = np.zeros((6, len(y)), dtype=float)
        y_new = y.copy()
        dy4 = np.zeros_like(y)
        dy5 = np.zeros_like(y)

        # Compute intermediate results
        for i in range(6):
            y_k = y.copy()
            for j in range(i):
                n = i*6+j
                y_k += self.coeffs[n]*k[j]
            k[i, :] = dt*func(t+self.t_coeffs[i]*dt, y_k)

        # Obtain 4th and 5th order solutions
        for i in range(6):
            dy4 += self.coeffs[36+i]*k[i]
            dy5 += self.coeffs[42+i]*k[i]

        # Compute absolute error in solution
        e = np.abs(dy5 - dy4)

        # Return result (based on 4th order solution!)
        y_new += dy4
        return y_new, e

    def adaptive_solver(self, func, y, tmax, dt, solout=None):

        t = 0.0
        i = 0
        i_max = 100     # Max allowable iterations per step
        N_out = 10
        rtol = self.rtol
        atol = self.atol

        # Yield initial values
        yield 0, t, y

        t_out = np.linspace(0.0, 1.1*tmax, N_out)
        t_out_next = t_out[0]

        # Integrate until tmax
        while t < tmax:
            i += 1
            y_new, e = self.rk45_step(func, y, t, dt)
            # Calculate tolerance ratio
            e_ratio = np.max(e/(rtol*np.abs(y_new) + atol)) + 1e-12
            # If NaN is encountered, reduce dt and
            # try again
            if np.isnan(e_ratio):
                print("NaN encountered")
                dt *= 0.1
                continue

            # In case that e_ratio oscillates around 1, break
            # loop and set to 1. The time step will be suboptimal
            # (too small), but that's ok
            if i > i_max and e_ratio < 1:
                e_ratio = 1

            # If error is within factor 10**0.1 from rtol
            # accept result and return
            if fabs(log10(e_ratio)) < 0.1:
                # If time step is larger than max time step
                dtmax = self.dtmax
                if dt > dtmax:
                    dt = dtmax
                    y, _ = self.rk45_step(func, y, t, dt)
                else:
                    y = y_new
                t += dt
                if t > t_out_next:
                    ind_next = bisect_right(t_out, t)
                    t_out_next = t_out[ind_next]
                    # print("[%i / %i] \t %.3e" % (ind_next, N_out, t))
                # print("Success dt: %3e" % dt)
                yield i, t, y
                if solout is not None:
                    solout(t, y, dt)
                i = 0
            # If error is too far from rtol, modify dt and
            # try again
            else:
                # Error is too large
                if e_ratio >= 1:
                    power = -0.2
                # Error is too
                elif e_ratio < 1:
                    power = -0.25
                # Modify dt
                s = e_ratio ** power
                dt *= s

        # Final call to solout
        if solout is not None:
            solout(t, y, dt)
        return False
