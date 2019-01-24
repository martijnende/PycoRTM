import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, j0, j1
import seaborn
from src.polar_solver import FD_solver


class AnalyticalSolution:
    """
    Analytical solution of a cooling sphere, subject to:
        T(r, 0) = T0
        T(r > R, t) = 0
    with no additional heat production or convection
    """

    def __init__(self, alpha, R, T0):
        self.alpha = alpha
        self.R = R
        self.T0 = T0
        # Compute the first 100 roots of the J0 bessel function
        self.ln = jn_zeros(0, 100) / R
        pass

    def compute_T(self, r, t):

        R = self.R
        ln = self.ln
        alpha = self.alpha

        T = 0.0

        for l in ln:
            T += j0(l*r) * np.exp(-alpha*t*l**2) / (l * j1(l*R))

        T *= 2*self.T0 / R
        return T


""" Basic problem parameters """

R = 40e3        # Radius [m]
N = 40          # Number of mesh nodes in r
alpha = 1.0e-4  # Thermal diffusivity [m2/s]

T0 = 400.0      # Initial temperature [K]
T_ext = 300.0   # External temperature [K]

t_yr = 3600*24*365.25   # Number of seconds in 1 year

# Mesh node positions
r = np.linspace(0, R-R/N, N)

# Mirror mesh (for symmetric plotting), set scale to km
r_both = np.hstack([-r[::-1], r]) * 1e-3

# Instantiate analytical solution class
ana = AnalyticalSolution(alpha, R, T0-T_ext)


""" Numerical simulation parameters """

N_r = N             # Number of mesh elements in r
N_theta = 50        # Number of mesh elements in theta

# Note that most parameters (related to convection) are set do zero
params = {
    "N_r": N_r,
    "N_theta": N_theta,
    "R": R,
    "o": 0.0,
    "alpha": alpha,
    "beta": 0.0,
    "dzeta": 0.0,
    "T_ext": T_ext,
    "t_half": 1.0,
    "N_IO": 1000,
    "N_print": 100,
    "rtol": 1e-10,      # High integration precision
    "atol": 1e-10,
}

solver = FD_solver(IO_file="cooling_benchmark_output.csv")
solver.set_params(params)

T0 = np.ones(N_r*N_theta)*T0        # Initial temperature vector
tmax = 50e3 * solver.t_yr           # Simulation duration
# solver.run_simulation(T0, tmax)     # Run simulation
data = solver.read()                # Read simulation output data

t_sim = np.unique(data["t"])        # Simulation time data
r_sim = np.unique(data["r"])        # Simulation mesh node positions

# Mirror and scale mesh for plotting
r_sim = np.hstack([-r_sim[::-1], r_sim]) * 1e-3

# Extract and mirror simulation temperature
T_sim = np.array(data["T"][data["theta"] == 0]).reshape((-1, N_r))
T_sim = np.hstack([T_sim[:, ::-1], T_sim])

# Nice colours
colours = seaborn.color_palette("gist_earth", len(t_sim))

# Loop over each simulation time step
for i, ti in enumerate(t_sim):
    # Compute analytical solution at time ti
    T = ana.compute_T(r, ti) + T_ext

    # Mirror and plot analytical solution
    T_both = np.hstack([T[::-1], T])
    plt.plot(r_both, T_both, c="k", ls="--")

    # Plot numerical solution
    plt.plot(r_sim, T_sim[i], c=colours[i])

# Insert a legend
plt.plot([], [], c="k", ls="--", label="Benchmark solution")
plt.plot([], [], c="k", ls="-", label="Numerical solution")
plt.legend(ncol=2, loc="bottom center")

plt.xlabel("radius [km]")
plt.ylabel("temperature [K]")

plt.tight_layout()
plt.show()
