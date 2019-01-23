import numpy as np
import pandas as pd


class IO:

    def __init__(self, IO_file):
        self.IO_file = IO_file
        self.IO_cols = ("t", "n", "r", "theta", "x", "y", "T", "u_r", "u_theta")
        pass

    def create_IO_file(self):
        df = pd.DataFrame(columns=self.IO_cols)
        with open(self.IO_file, "w") as f:
            df.to_csv(f, index=False)
        return True

    def read(self):
        print("Reading data")
        df = pd.read_csv(self.IO_file, names=self.IO_cols, header=0)
        self.data = df
        return df

    def write(self, t, T, u_r, u_theta):

        N_r = self.N_r
        N_theta = self.N_theta

        n = np.arange(N_r*N_theta)
        j = n % N_theta
        i = (n - j)/N_r

        r = i*self.dr
        theta = j*self.dtheta

        x = r*np.cos(theta)
        y = r*np.sin(theta)

        t = np.ones(N_r*N_theta) * t

        output_dict = {
            "t": t, "n": n, "r": r, "theta": theta, "x": x, "y": y,
            "T": T, "u_r": u_r, "u_theta": u_theta,
        }

        output_df = pd.DataFrame(output_dict, columns=self.IO_cols)
        with open(self.IO_file, "a") as f:
            output_df.to_csv(f, header=False, index=False)

        return True
