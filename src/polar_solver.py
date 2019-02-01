import numpy as np
from scipy.sparse import diags, eye, csc_matrix
from scipy.sparse.linalg import spsolve, inv
from src.IO_lib import IO
from src.plot_functions import animator
from src.rk45 import rk45_solver
from src.auxiliary import helper
from time import time


class FD_solver(animator, helper, IO, rk45_solver):

    def __init__(self):
        animator.__init__(self)
        helper.__init__(self)
        IO.__init__(self)
        rk45_solver.__init__(self)
        self.t_yr = 3600*24*365.25
        pass

    def set_params(self, params):
        self.__dict__.update(params)
        self.dr = self.R / self.N_r
        self.dtheta = self.theta_range / self.N_theta
        self.dx_min = min(self.dr, self.dr*self.dtheta)
        r = (np.arange(self.N_r) + 0.5) * self.dr
        theta = np.arange(self.N_theta) * self.dtheta
        R, Theta = np.meshgrid(r, theta)
        self.X = R * np.cos(Theta)
        self.Y = R * np.sin(Theta)

    def construct_kernels(self):
        N_r = self.N_r
        N_theta = self.N_theta
        dr = self.dr
        dtheta = self.dtheta

        alpha = self.alpha
        T_ext = self.T_ext

        N_t = N_r * N_theta
        i = np.arange(N_r) + 0.5
        i = np.repeat(i, N_theta)
        I = np.ones(N_t)
        i_max = N_theta * (N_r - 1)     # Exterior elements

        O = np.zeros(N_t)

        kernels = {
            "T": {},
            "psi": {},
        }

        """ Construct temperature kernels (Laplacian) """

        """ Construct T_r matrix """
        # Central first differences stencil
        a1 = -1 / (2 * i * dr ** 2) * I
        a2 = -a1

        # Interior B.C.: no gradient
        a1[:N_theta] = 0.0
        a2[:N_theta] = 0.0

        # Exterior B.C.: fixed T = T_ext
        O[i_max:] += alpha * a2[-1] * T_ext

        # Construct sparse matrix
        A = diags([a1[N_theta:], 0, a2[:-N_theta]], offsets=[-N_theta, 0, N_theta])

        """ Construct T_rr matrix """
        # Second differences stencil
        a1 = 1 / (dr ** 2) * I
        a2 = -2 * a1
        a3 = a1.copy()

        # Interior B.C.: no gradient --> x(i+1) = x(i-1)
        # --> x(i-1) - 2x(i) + x(i+1) = 2x(i-1) - 2x(i)
        a1[:N_theta] = 0.0
        a3[:N_theta] *= 2.0

        # Exterior B.C.: fixed T = T_ext
        O[i_max:] += alpha * a3[-1] * T_ext

        # Construct sparse matrix
        B = diags([a1[N_theta:], a2, a3[:-N_theta]], offsets=[-N_theta, 0, N_theta])

        """ Construct T_tt matrix """
        # Second differences stencil
        a1 = 1 / (i * dr * dtheta) ** 2 * I
        a2 = -2 * a1
        a3 = a1.copy()

        # Interior B.C.: identical elements
        a1[:N_theta] = 0.0
        a2[:N_theta] = 0.0
        a3[:N_theta] = 0.0

        # Left periodic B.C: T[i, -1] = T[i, N_theta-1]
        a4 = np.zeros(N_t)
        a4[::N_theta] = a1[::N_theta]
        a1[::N_theta] = 0.0

        # Right periodic B.C: T[i, N_theta] = T[i, 0]
        a5 = np.zeros(N_t)
        a5[N_theta - 1::N_theta] = a3[N_theta - 1::N_theta]
        a3[N_theta - 1::N_theta] = 0.0

        C = diags(
            [a5[N_theta - 1:], a1[1:], a2, a3[:-1], a4[:-N_theta + 1]],
            offsets=[-N_theta + 1, -1, 0, 1, N_theta - 1]
        )

        # Construct Laplacian = nabla^2 (note: not multiplied by alpha)
        laplacian_T = A + B + C
        kernels["T"]["laplacian"] = laplacian_T

        """ Construct temperature kernels (gradient) """

        """ Construct T_r matrix """
        # Central first differences stencil
        a1 = -1 / (2 * dr) * I
        a2 = -a1

        # Interior B.C.: no gradient
        a1[:N_theta] = 0.0
        a2[:N_theta] = 0.0

        # Construct sparse matrix
        T_r = diags([a1[N_theta:], 0, a2[:-N_theta]], offsets=[-N_theta, 0, N_theta])
        kernels["T"]["grad_r"] = T_r

        """ Construct T_theta matrix """
        # Central first differences stencil
        a1 = -1 / (2 * i * dr * dtheta) * I
        a2 = -a1

        # Interior B.C.: identical elements
        a1[:N_theta] = 0.0
        a2[:N_theta] = 0.0

        # Left periodic B.C: T[i, -1] = T[i, N_theta-1]
        a3 = np.zeros(N_t)
        a3[::N_theta] = a1[::N_theta]
        a1[::N_theta] = 0.0

        # Right periodic B.C: T[i, N_theta] = T[i, 0]
        a4 = np.zeros(N_t)
        a4[N_theta - 1::N_theta] = a2[N_theta - 1::N_theta]
        a2[N_theta - 1::N_theta] = 0.0

        # Construct sparse matrix
        T_theta = diags(
            [a4[N_theta - 1:], a1[1:], a2[:-1], a3[:-N_theta + 1]],
            offsets=[-N_theta + 1, -1, 1, N_theta - 1]
        )
        kernels["T"]["grad_theta"] = T_theta

        """ Construct r x T_theta matrix """
        # Central first differences stencil
        a1 = -1 / (2 * dtheta) * I
        a2 = -a1

        # Interior B.C.: identical elements
        a1[:N_theta] = 0.0
        a2[:N_theta] = 0.0

        # Left periodic B.C: T[i, -1] = T[i, N_theta-1]
        a3 = np.zeros(N_t)
        a3[::N_theta] = a1[::N_theta]
        a1[::N_theta] = 0.0

        # Right periodic B.C: T[i, N_theta] = T[i, 0]
        a4 = np.zeros(N_t)
        a4[N_theta - 1::N_theta] = a2[N_theta - 1::N_theta]
        a2[N_theta - 1::N_theta] = 0.0

        # Construct sparse matrix
        r_T_theta = diags(
            [a4[N_theta - 1:], a1[1:], a2[:-1], a3[:-N_theta + 1]],
            offsets=[-N_theta + 1, -1, 1, N_theta - 1]
        )
        kernels["T"]["r_grad_theta"] = r_T_theta

        """ Construct stream function kernels (Laplacian) """

        """ Construct psi_r / r matrix """
        # Central first differences stencil
        a1 = -1 / (2 * i * dr ** 2) * I
        a2 = -a1
        a3 = np.zeros_like(a1)

        # Interior B.C.: no flow
        a1[:N_theta] *= 0
        a3[:N_theta] = -1

        # Exterior B.C.: free surface (u_th^N+1 = u_th^N)
        a1[i_max:] *= -1
        a2[i_max:] *= 0
        a3[i_max:] = 1

        # Construct sparse matrix
        # A = diags([a1[N_theta:], 0, a2[:-N_theta]], offsets=[-N_theta, 0, N_theta])
        A = diags([a1[N_theta:], a3, a2[:-N_theta]], offsets=[-N_theta, 0, N_theta])

        """ Construct psi_rr matrix """
        # Second differences stencil
        a1 = 1 / (dr ** 2) * I
        a2 = -2 * a1
        a3 = a1.copy()

        # Interior B.C.: no gradient --> x(i+1) = x(i-1)
        # --> x(i-1) - 2x(i) + x(i+1) = 2x(i+1) - 2x(i)
        a1[:N_theta] *= 0
        a2[:N_theta] *= (3.0 / 2.0)
        a3[:N_theta] *= 3

        # Exterior B.C.: free surface (u_th^N+1 = u_th^N)
        a1[i_max:] *= 3
        a2[i_max:] *= (3.0/2.0)
        a3[i_max:] *= 0

        # Construct sparse matrix
        B = diags([a1[N_theta:], a2, a3[:-N_theta]], offsets=[-N_theta, 0, N_theta])

        """ Construct psi_tt matrix """
        # Second differences stencil
        a1 = 1 / (i * dr * dtheta) ** 2 * I
        a2 = -2 * a1
        a3 = a1.copy()

        # Interior B.C.: identical elements
        a1[:N_theta] = 0.0
        a2[:N_theta] = 0.0
        a3[:N_theta] = 0.0

        # Exterior B.C.: no radial flow for all j, which
        # implies that d^2 psi / d theta^2 = 0 (psi uniform)
        a1[i_max:] = 0.0
        a2[i_max:] = 0.0
        a3[i_max:] = 0.0


        # Left periodic B.C: T[i, -1] = T[i, N_theta-1]
        a4 = np.zeros(N_t)
        a4[::N_theta] = a1[::N_theta]
        a1[::N_theta] = 0.0

        # Right periodic B.C: T[i, N_theta] = T[i, 0]
        a5 = np.zeros(N_t)
        a5[N_theta - 1::N_theta] = a3[N_theta - 1::N_theta]
        a3[N_theta - 1::N_theta] = 0.0

        C = diags(
            [a5[N_theta - 1:], a1[1:], a2, a3[:-1], a4[:-N_theta + 1]],
            offsets=[-N_theta + 1, -1, 0, 1, N_theta - 1]
        )

        # Construct Laplacian = nabla^2 (note: not multiplied by alpha)
        laplacian_psi = A + B + C

        kernels["psi"]["laplacian"] = laplacian_psi

        """ Construct psi kernels (gradient) """

        """ Construct psi_r matrix """
        # Central first differences stencil
        a1 = -1 / (2 * dr) * I
        a2 = -a1
        a3 = np.zeros_like(a1)

        # Interior B.C.: no gradient
        a1[:N_theta] = 0.0
        a2[:N_theta] = 0.0

        # Exterior B.C.: no gradient
        a1[i_max:] = 0.0
        a2[i_max:] = 0.0

        # Construct sparse matrix
        psi_r = diags([a1[N_theta:], a3, a2[:-N_theta]], offsets=[-N_theta, 0, N_theta])
        kernels["psi"]["grad_r"] = psi_r

        """ Construct psi_theta matrix """
        # Central first differences stencil
        a1 = -1 / (2 * i * dr * dtheta) * I
        a2 = -a1

        # Interior B.C.: identical elements
        a1[:N_theta] = 0.0
        a2[:N_theta] = 0.0
        #
        # Exterior B.C.: no gradients
        a1[i_max:] = 0.0
        a2[i_max:] = 0.0

        # Left periodic B.C: T[i, -1] = T[i, N_theta-1]
        a3 = np.zeros(N_t)
        a3[::N_theta] = a1[::N_theta]
        a1[::N_theta] = 0.0

        # Right periodic B.C: T[i, N_theta] = T[i, 0]
        a4 = np.zeros(N_t)
        a4[N_theta - 1::N_theta] = a2[N_theta - 1::N_theta]
        a2[N_theta - 1::N_theta] = 0.0

        # Construct sparse matrix
        psi_theta = diags(
            [a4[N_theta - 1:], a1[1:], a2[:-1], a3[:-N_theta + 1]],
            offsets=[-N_theta + 1, -1, 1, N_theta - 1]
        )
        kernels["psi"]["grad_theta"] = psi_theta

        # Store source function (which includes B.C.)
        kernels["O"] = O

        self.kernels = kernels

    def prepare_ODE_fixed_dt(self, K, dt):
        I = eye(K.shape[0])
        K = 0.5 * dt * K
        A1 = I - K
        A0 = I + K
        A1_inv = inv(csc_matrix(A1))
        self.A = A1_inv.dot(A0)

    def do_step_fixed(self, T, O, dt):
        return self.A.dot(T) + dt*O

    def do_step(self, t, T, dt):

        kernels = self.kernels

        # Step 1: compute r_T_theta, solve for psi
        r_T_theta = kernels["T"]["r_grad_theta"].dot(T)
        psi = spsolve(kernels["psi"]["laplacian"], self.dzeta * r_T_theta, permc_spec="MMD_AT_PLUS_A")

        # Step 2: compute velocity components = grad psi
        u_r = kernels["psi"]["grad_theta"].dot(psi)
        u_theta = -kernels["psi"]["grad_r"].dot(psi)

        # Step 3: compute diffusive and advective terms
        dT_diff = self.alpha * kernels["T"]["laplacian"].dot(T)
        T_r = kernels["T"]["grad_r"].dot(T)
        T_theta = kernels["T"]["grad_theta"].dot(T)
        dT_adv = self.beta * (u_r * T_r + u_theta * T_theta)

        # Step 4: compute source function (including B.C.)
        O_base = kernels["O"]
        O = self.o * np.exp(-t / self.t_half)

        # Step 4: compute next T using forward Euler
        T_new = T + dt*(dT_diff + dT_adv + O_base + O)

        return T_new, u_r, u_theta

    def RK_solout(self, t, T, dt):

        self.i += 1
        self.t = t

        if (self.i % self.N_IO_time == 0) or (self.i % self.N_IO_snap == 0) or (self.i % self.N_print == 0):
            kernels = self.kernels

            # Step 1: compute r_T_theta, solve for psi
            r_T_theta = kernels["T"]["r_grad_theta"].dot(T)
            psi = spsolve(kernels["psi"]["laplacian"], self.dzeta * r_T_theta)

            # Step 2: compute velocity components = grad psi
            u_r = kernels["psi"]["grad_theta"].dot(psi)
            u_theta = -kernels["psi"]["grad_r"].dot(psi)

            if self.i % self.N_print == 0:
                self.screen_write(t, dt, T, u_r, u_theta)
            if (self.i % self.N_IO_time == 0) and (self.IO_file_time is not None):
                self.OT_file_write(t, T)
            if (self.i % self.N_IO_snap == 0) and (self.IO_file_snap is not None):
                self.OX_file_write(t, T, u_r, u_theta)

        pass

    def T_dot(self, t, T):

        if (any(T < self.T_ext)):
            print("T - T_ext < 0!")
            print("The internal temperature dropped below the external temperature")
            print("This is probably a resolution problem. Increase resolution")
            exit()

        kernels = self.kernels
        dt = t - self.t
        N_r, N_theta = self.N_r, self.N_theta

        n = np.arange(N_r * N_theta)
        j = n % N_theta
        i = (n - j) / N_theta + 0.5
        r = i * self.dr

        # Step 1: compute r_T_theta, solve for psi
        r_T_theta = kernels["T"]["r_grad_theta"].dot(T)
        psi = spsolve(kernels["psi"]["laplacian"], self.dzeta * r_T_theta)

        # Step 2: compute velocity components = grad psi
        u_r = kernels["psi"]["grad_theta"].dot(psi)
        u_theta = -kernels["psi"]["grad_r"].dot(psi)

        # Divergence terms of u
        div_u_r = u_r / r + kernels["psi"]["grad_r"].dot(u_r)
        div_u_theta = kernels["psi"]["grad_theta"].dot(u_theta)

        # Find limiting time step due to advection
        dt_r = np.nanmin(self.dr / np.abs(u_r))
        dt_theta = np.nanmin(r * self.dtheta / np.abs(u_theta))
        dt_crit = np.min([dt_r, dt_theta])

        if dt > 0.9*dt_crit:
            self.dtmax = 0.9*dt_crit

        # Step 3: compute diffusive and advective terms
        dT_diff = self.alpha * kernels["T"]["laplacian"].dot(T)
        T_r = kernels["T"]["grad_r"].dot(T)
        T_theta = kernels["T"]["grad_theta"].dot(T)

        # Advective term includes non-solenoidal flow (div u =/= 0),
        # owing to limited numerical accuracy.
        dT_adv1 = self.beta * (u_r * T_r + u_theta * T_theta)
        dT_adv2 = self.beta * T * (div_u_r + div_u_theta)
        dT_adv = dT_adv1 + dT_adv2

        # Step 4: compute source function (including B.C.)
        O_base = kernels["O"]
        O = self.o * np.exp(-t / self.t_half)

        # Step 4: compute next T using forward Euler
        T_dot = dT_diff - dT_adv + O_base + O

        return T_dot

    def run_simulation(self, T, tmax):

        self.construct_kernels()
        print("Kernels constructed")

        self.create_IO_files()
        print("Output file created")

        print("Running simulation...")
        self.t = 0.0
        self.tmax = tmax
        self.dtmax = tmax
        self.i = 0
        self.t_wall_start = time()

        dt0 = 1e3
        gen = self.adaptive_solver(
            func=self.T_dot, y=T, tmax=tmax, dt=dt0, solout=self.RK_solout
        )
        i, t, y = zip(*gen)

        print("Done")

        return True
