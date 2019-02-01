import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.polar_solver import FD_solver
from scipy.sparse.linalg import spsolve


def compute_u(T, kernels, dzeta):
    r_T_theta = kernels["T"]["r_grad_theta"].dot(T)
    x = dzeta*r_T_theta
    psi = spsolve(kernels["psi"]["laplacian"], x)
    # psi /= np.nanmax(psi)

    # Step 2: compute velocity components = grad psi
    u_r = kernels["psi"]["grad_theta"].dot(psi)
    u_theta = -kernels["psi"]["grad_r"].dot(psi)

    return psi, u_r, u_theta

""" Benchmark test parameters """

R = 10e3            # Radius [m]
N_r = 45           # Number of mesh nodes in r
N_theta = 45       # Number of mesh nodes in theta

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
    "IO_file_time": None,
    "IO_file_snap": "snap_output_Nr=45_Nt=45.csv",
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
r = (i+0.5)*dr
theta = j*dtheta
inds_margin = (i == N_r-1)

dzeta = 1e-6

# Construct temperature function: T = f(r, theta)
T0 = 100
T_ext = 10*np.pi

# Case 1: only radial gradient in T = no convection
T = T0 + (T_ext - T0)*(r / R)**2
psi, u_r, u_theta = compute_u(T, kernels, dzeta)

u_r_ok = np.allclose(u_r, 0.0, rtol=1e-8)
u_theta_ok = np.allclose(u_theta, 0.0, rtol=1e-8)

if u_r_ok and u_theta_ok:
    print("Radial temperature gradients ok")
else:
    print("Radial temperature gradients not ok")

# Case 2: only azimuthal gradient in T
# T = 2*T*np.cos(theta)*np.sin(r/R) + 0*100*np.random.rand(N_r*N_theta)
# inds_block = (i > 30) & (i < 39) & (j > 10) & (j < 20)
# inds_block2 = (i > 25) & (i < 39) & (j > 30) & (j < 45)
# T[inds_block] = 2*T0
# T[inds_block2] = 2*T0
solver.read()
T = solver.ox_data["T"].values.reshape(-1, N_r, N_theta)
print(T)
psi, u_r, u_theta = compute_u(T, kernels, dzeta)

# Compute divergence of flux (should be zero)
mag_u = np.sqrt(u_r*u_r + u_theta*u_theta)
div_u = (1.0/r) * kernels["psi"]["grad_r"].dot(r*u_r) + kernels["psi"]["grad_theta"].dot(u_theta)
du_r = kernels["psi"]["grad_r"].dot(u_r)
du_theta = kernels["psi"]["grad_theta"].dot(u_theta)
psi_rtheta = kernels["psi"]["grad_theta"].dot(kernels["psi"]["grad_r"].dot(psi))

plt.subplot(321)
plt.imshow(T.reshape((N_r, N_theta)), cmap="coolwarm", origin="lower")
plt.subplot(322)
plt.imshow(psi.reshape((N_r, N_theta)), cmap="coolwarm", origin="lower")
plt.subplot(323)
plt.imshow(u_r.reshape((N_r, N_theta)), cmap="coolwarm", origin="lower")
plt.subplot(324)
plt.imshow(u_theta.reshape((N_r, N_theta)), cmap="coolwarm", origin="lower")
plt.subplot(325)
plt.imshow((div_u/mag_u).reshape((N_r, N_theta)), cmap="coolwarm", origin="lower")
plt.tight_layout()
plt.show()
# exit()



# plt.plot(psi_rtheta)
# plt.plot(du_r)
# # plt.plot(du_theta)
# plt.show()
# exit()

mag_u = np.sqrt(u_r*u_r + u_theta*u_theta)
# div_u[np.isnan(div_u)] = 0
# print(np.allclose(div_u, 0.0, rtol=1e-1))
# print(np.median(np.power(div_u, 2)))
# plt.plot(u_r.reshape((N_r, N_theta))[:, 0])
plt.plot(u_r)
plt.plot(u_theta)
plt.twinx()
plt.plot(div_u, c="C2", alpha=0.2)
# plt.twinx()
# plt.plot(mag_u, c="C1")
plt.show()
exit()


# Check that no radial flow exists (boundary condition)
ur_margin = u_r[inds_margin]
ur_margin_ok = np.allclose(ur_margin, 0.0, rtol=1e-8)

u_r_ok = np.allclose(u_r, 0.0, rtol=1e-8)
u_theta_ok = np.allclose(u_theta, 0.0, rtol=1e-8)

# Extract cartesian coordinates
X = np.vstack([solver.X, solver.X[0]])
Y = np.vstack([solver.Y, solver.Y[0]])

u_r = u_r.reshape((N_r, N_theta)).T
u_r = np.vstack([u_r, u_r[0]])

u_theta = u_theta.reshape((N_r, N_theta)).T
u_theta = np.vstack([u_theta, u_theta[0]])

psi = psi.reshape((N_r, N_theta)).T
psi = np.vstack([psi, psi[0]])

T = T.reshape((N_r, N_theta)).T
T = np.vstack([T, T[0]])

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, psi, cmap="coolwarm")
plt.tight_layout()

plt.figure(2, figsize=(16, 4))

plt.subplot(141, aspect="equal")
plt.contourf(X, Y, T, 100, cmap="coolwarm")
plt.axis("off")
plt.xlim((-1.01 * R, 1.01 * R))
plt.ylim((-1.01 * R, 1.01 * R))
plt.title("Temperature")

plt.subplot(142, aspect="equal")
plt.contourf(X, Y, psi, 100, cmap="coolwarm")
CS = plt.contour(X, Y, psi, 5)
plt.clabel(CS, inline=1, fontsize=10)
plt.axis("off")
plt.xlim((-1.01 * R, 1.01 * R))
plt.ylim((-1.01 * R, 1.01 * R))
plt.title(r"$\psi$")

plt.subplot(143, aspect="equal")
plt.contourf(X, Y, u_r, 100, cmap="coolwarm")
CS = plt.contour(X, Y, u_r, 5)
plt.clabel(CS, inline=1, fontsize=10)
plt.axis("off")
plt.xlim((-1.01 * R, 1.01 * R))
plt.ylim((-1.01 * R, 1.01 * R))
plt.title("Radial flow velocity")

plt.subplot(144, aspect="equal")
plt.contourf(X, Y, u_theta, 100, cmap="coolwarm")
CS = plt.contour(X, Y, u_theta, 5)
plt.clabel(CS, inline=1, fontsize=10)
plt.axis("off")
plt.xlim((-1.01 * R, 1.01 * R))
plt.ylim((-1.01 * R, 1.01 * R))
plt.title("Azimuthal flow velocity")

plt.tight_layout()
plt.show()

