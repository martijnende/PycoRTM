import numpy as np
from src.polar_solver import FD_solver

""" Physical constants """

t_yr = 3600*24*365.25       # Seconds per year
t_half = 7.17e5 * t_yr      # 26Al half-life [s]
E = 1.487e13                # Radioactive decay heat release [J/kg]
f = 4.5e-8                  # Mass fraction of 26Al [kg/kg]
phi = 0.4                   # Porosity [-]
rho_f = 1000                # Fluid density [kg/m3]
cp_f = 4185.5               # Fluid heat capacity [J/kg/K]
rho_r = 3000                # Rock density [kg/m3]
cp_r = 900                  # Rock heat capacity [J/kg/K]
kappa = 2.0                 # Thermal conductivity [J/s/m/K]
G = 6.67408e-11             # Gravitational constant [m3/kg/s2]
lam = 2.07e-4               # Thermal expansion coefficient [1/K]
eta = 8.9e-4                # Dynamic viscosity [Pa s]
k = 1e-12                   # Fluid permeability [m2]

# Bulk density [kg/m3]
rho_b = (1-phi)*rho_r + phi*rho_f
# Average specific heat [J/m3/K]
rho_cp = (1-phi)*rho_r*cp_r + phi*rho_f*cp_f
# Thermal diffusivity [m2/s]
alpha = kappa / rho_cp
# Advection constant [-]
beta = rho_f*cp_f / rho_cp
# Partial gravitational acceleration [1/s2]
gamma = (4.0/3.0)*np.pi*rho_b*G
# Buoyancy constant [1/K/s]
dzeta = k*rho_f*lam*gamma / (eta*phi)
# Radiactive decay heat constant
o = (1 - phi)*rho_r*f*E / (t_half*rho_cp)  # Decay heat release constant [J/m3/s]

""" Mesh parameters """
N_r = 40
N_theta = 50

params = {
    "N_r": N_r,
    "N_theta": N_theta,
    "R": 100e3,
    "o": o,
    "alpha": alpha,
    "beta": beta,
    "dzeta": dzeta,
    "T_ext": 10.0,
    "t_half": t_half,
    "N_IO": 10,
    "N_print": 10,
    "rtol": 1e-6,
    "atol": 1e-8,
}
solver = FD_solver(IO_file="output.csv")
solver.set_params(params)

# Rayleigh number (-]
A = rho_cp * o / rho_r
Ra = A*rho_f*k*lam*gamma*params["R"]**4 / (3*alpha**2 * cp_r * eta)
print("Rayleigh number: %.3e" % Ra)
print("Critical Rayleigh number (Young et al): 183.91")

i = np.arange(int(0.3*N_r), int(0.6*N_r))
j = np.arange(int(0.0*N_theta), int(0.2*N_theta))
I, J = np.meshgrid(i, j)
n = I*N_theta + J

T0 = np.ones(N_r*N_theta)*300 + 10*np.random.rand(N_r*N_theta)
# T0[n] = 300

tmax = 3e6 * solver.t_yr * 1e6

# solver.run_simulation(T0, tmax)
data = solver.read()
# solver.animate(data, T_range=(300, 700), v_range=(-10, -9))

t = data["t"].unique()
inds = data["t"] == t[-1]
# solver.plot_frame(inds, T_range=(300, 700), v_range=(-10, -9))


