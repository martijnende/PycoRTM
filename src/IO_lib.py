import numpy as np
import pandas as pd
from time import time


class IO:

    def __init__(self):
        # Time series file
        self.OT_cols = ("t", "T_mean", "T_max")

        # Snapshot file
        self.OX_cols = ("t", "n", "r", "theta", "x", "y", "T", "u_r", "u_theta")
        pass

    def create_IO_files(self):

        if self.IO_file_time is not None:
            df = pd.DataFrame(columns=self.OT_cols)
            with open(self.IO_file_time, "w") as f:
                df.to_csv(f, index=False)

        if self.IO_file_snap is not None:
            df = pd.DataFrame(columns=self.OX_cols)
            with open(self.IO_file_snap, "w") as f:
                df.to_csv(f, index=False)
        return True

    def read(self):

        if self.IO_file_time is not None:
            print("Reading time series data")
            df = pd.read_csv(self.IO_file_time, names=self.OT_cols, header=0)
            self.ot_data = df

        if self.IO_file_snap is not None:
            print("Reading snapshot data")
            df = pd.read_csv(self.IO_file_snap, names=self.OX_cols, header=0)
            self.ox_data = df
        return True

    def screen_write(self, t, dt, T, u_r, u_theta):
        T_mean = self.compute_mean(T)
        t_elapsed = time() - self.t_wall_start
        tmax = self.tmax
        ETA = (t_elapsed / t) * (tmax - t) / 60
        print(
            "t = %.1f kyr \t dt = %.2e yr \t Tmax = %.1f K \t Tmean = %.1f K \t vmax = %.3e m/s \t ETA = %.1f min" %
            (1e-3 * t / self.t_yr, dt / self.t_yr, T.max(), T_mean,
             np.max(np.sqrt(u_r ** 2 + u_theta ** 2)), ETA)
        )
        return True

    def OT_file_write(self, t, T):

        T_max = T.max()
        T_mean = self.compute_mean(T)

        output_dict = {
            "t": t, "T_mean": T_mean, "T_max": T_max,
        }

        output_df = pd.DataFrame(output_dict, columns=self.OT_cols, index=[0])
        with open(self.IO_file_time, "a") as f:
            output_df.to_csv(f, header=False, index=False)

        return True

    def OX_file_write(self, t, T, u_r, u_theta):

        N_r = self.N_r
        N_theta = self.N_theta

        n = np.arange(N_r*N_theta)
        j = n % N_theta
        i = (n - j)/N_theta + 0.5

        r = i*self.dr
        theta = j*self.dtheta

        x = r*np.cos(theta)
        y = r*np.sin(theta)

        t = np.ones(N_r*N_theta) * t

        output_dict = {
            "t": t, "n": n, "r": r, "theta": theta, "x": x, "y": y,
            "T": T, "u_r": u_r, "u_theta": u_theta,
        }

        output_df = pd.DataFrame(output_dict, columns=self.OX_cols)
        with open(self.IO_file_snap, "a") as f:
            output_df.to_csv(f, header=False, index=False)

        return True
