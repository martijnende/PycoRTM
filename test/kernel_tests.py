import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.polar_solver import FD_solver


""" Benchmark test parameters """

R = 10e3            # Radius [m]
N_r = 100           # Number of mesh nodes in r
N_theta = 200       # Number of mesh nodes in theta

params = {
    "N_r": N_r,
    "N_theta": N_theta,
    "R": R,
    "o": 0.0,
    "alpha": 0.0,
    "beta": 0.0,
    "dzeta": 0.0,
    "T_ext": 0.0,
    "t_half": 1.0,
    "N_IO": 0,
    "N_print": 0,
    "rtol": 1e-10,
    "atol": 1e-10,
}

solver = FD_solver()
solver.set_params(params)
solver.construct_kernels()

# Extract kernels (differentiation matrices)
kernels = solver.kernels

# Extract cartesian coordinates
X = np.vstack([solver.X, solver.X[0]])
Y = np.vstack([solver.Y, solver.Y[0]])

# Mesh element ordering
n = np.arange(N_r*N_theta)
j = n % N_theta
i = (n - j)/N_theta

# Mesh resolution
dr = R / N_r
dtheta = 2*np.pi / N_theta

# Cylindrical coordinates of mesh nodes
r = i*dr
theta = j*dtheta

# Precompute common factor
r_R = 1 - r/R

# Construct temperature function: T = f(r, theta)
T0 = 2.0
T = r_R**2 * np.cos(theta) * T0

""" Compute gradients """

# Gradient in r
T_r = -(2/R)*r_R * np.cos(theta) * T0
T_r_kernel = kernels["T"]["grad_r"].dot(T)

# Gradient in theta (note: gradient in cylindrical coordinates!)
T_theta = -r_R**2 * np.sin(theta) * T0 / r
T_theta_kernel = kernels["T"]["grad_theta"].dot(T)

# Gradient in theta, multiplied by r
rT_theta = -r_R**2 * np.sin(theta) * T0
rT_theta_kernel = kernels["T"]["r_grad_theta"].dot(T)

# Laplacian operator (note: cylindrical coordinates!)
T_laplace = np.cos(theta) * T0 * (2/R**2 - 2*r_R/(r*R) - r_R**2 / r**2)
T_laplace_kernel = kernels["T"]["laplacian"].dot(T)

# Structure results
results = {
    r"$\nabla_r T$": {
        "analytical": T_r,
        "numerical": T_r_kernel,
        "range": (-2*T0/R, 2*T0/R),
    },
    r"$\nabla_{\theta} T$": {
        "analytical": T_theta,
        "numerical": T_theta_kernel,
        "range": (-2*T0/R, 2*T0/R),
    },
    r"$r \times \nabla_{\theta} T$": {
        "analytical": rT_theta,
        "numerical": rT_theta_kernel,
        "range": (-T0, T0),
    },
    r"$\nabla^2 T$": {
        "analytical": T_laplace,
        "numerical": T_laplace_kernel,
        "range": (-2*T0/R**2, 2*T0/R**2),
    },
}

result_list = (r"$\nabla_r T$", r"$\nabla_{\theta} T$", r"$r \times \nabla_{\theta} T$", r"$\nabla^2 T$")
N_results = len(result_list)


""" Compute gradients """

plt.figure(figsize=(4*N_results, 8))

for i, key in enumerate(result_list):
    result = results[key]
    ana_res = result["analytical"].reshape((N_r, N_theta)).T
    ana_res = np.vstack([ana_res, ana_res[0]])
    num_res = result["numerical"].reshape((N_r, N_theta)).T
    num_res = np.vstack([num_res, num_res[0]])

    amin, amax = result["range"]
    levels = np.linspace(amin, amax, 200)

    circle = matplotlib.patches.Circle(
        xy=[0, 0], radius=(N_r - 1) * dr,
        edgecolor="k", facecolor="none", alpha=0.5
    )

    circle2 = matplotlib.patches.Circle(
        xy=[0, 0], radius=(N_r - 1) * dr,
        edgecolor="k", facecolor="none", alpha=0.5
    )

    ax = plt.subplot(2, N_results, 1+i, aspect="equal")
    plt.contourf(X, Y, ana_res, levels, cmap="coolwarm", vmin=amin, vmax=amax)
    ax.add_patch(circle)
    plt.axis("off")
    plt.xlim((-1.01 * R, 1.01 * R))
    plt.ylim((-1.01 * R, 1.01 * R))
    plt.title(r"Analytical %s" % key)

    ax = plt.subplot(2, N_results, 1+N_results+i, aspect="equal")
    plt.contourf(X, Y, num_res, levels, cmap="coolwarm", vmin=amin, vmax=amax)
    ax.add_patch(circle2)
    plt.axis("off")
    plt.xlim((-1.01 * R, 1.01 * R))
    plt.ylim((-1.01 * R, 1.01 * R))
    plt.title(r"Numerical %s" % key)

plt.tight_layout()
plt.show()

