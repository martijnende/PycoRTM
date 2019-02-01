import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


class animator:

    def __init__(self):
        pass

    def init_plot(self):
        pass

    def update_animation(self, i, ax, X, Y, T, cmap, Tmin, Tmax):
        for c in self.CS.collections:
            c.remove()
        self.CS = plt.contourf(
            X, Y, T[:, :, i], 200, cmap=cmap, vmin=Tmin, vmax=Tmax
        )
        return self.CS

    def update_animation_save(self, t):
        ani_dict = self.ani_dict
        X, Y, T, v = ani_dict["X"], ani_dict["Y"], ani_dict["T"], ani_dict["v"]
        R = self.R
        cmap = ani_dict["cmap"]
        Tmin, Tmax = ani_dict["T_range"]
        vmin, vmax = ani_dict["v_range"]
        tmax = ani_dict["tmax"]
        tmax_real = ani_dict["tmax_real"]
        ax1, ax2 = ani_dict["axes"]

        i = int(t*self.N_t / tmax)
        timer = "Time: %.3f Myr" % (tmax_real * t / (1e6 * tmax * self.t_yr))

        # Clear previous contour plot
        self.timer.remove()
        for c in self.CS1.collections:
            c.remove()
        for c in self.CS2.collections:
            c.remove()

        # Redraw contour plot with current data
        fig = self.fig
        self.timer = plt.text(-R, R, timer)
        self.CS1 = ax1.contourf(
            X, Y, T[:, :, i], 200, cmap=cmap, vmin=Tmin, vmax=Tmax
        )
        self.CS2 = ax2.contourf(
            X, Y, v[:, :, i], 200, cmap=cmap, vmin=vmin, vmax=vmax
        )

        return mplfig_to_npimage(self.fig)

    def animate(self, data, T_range=(None, None), v_range=(None, None)):

        N_r = self.N_r
        N_theta = self.N_theta
        R = self.R
        t_vals = data["t"].unique()
        dt = np.hstack([0, np.diff(t_vals)])
        self.N_t = len(t_vals)

        n = np.arange(N_r * N_theta)
        j = n % N_theta
        i = (n - j) / N_theta + 0.5
        r = i * self.dr
        N_t = len(t_vals)

        # Map data for theta = 0 to theta = 2*pi
        # to make the contour plots periodic
        X = np.vstack([self.X, self.X[0]])
        Y = np.vstack([self.Y, self.Y[0]])

        T_old = data["T"].values.reshape(-1, N_r*N_theta)
        T = data["T"].values.reshape(-1, X.shape[1], X.shape[0]-1).T
        T = np.concatenate([T, T[0, :, :].reshape(1, T.shape[1], len(t_vals))], axis=0)
        T0 = T[:, :, 0]

        # v = np.sqrt(data["u_r"]**2 + data["u_theta"]**2)
        # v = np.log10(v.values.reshape(-1, X.shape[1], X.shape[0] - 1).T)
        # v = v.values.reshape(-1, X.shape[1], X.shape[0]-1).T
        kernels = self.kernels
        u_r = data["u_r"].values.reshape(N_t, N_r*N_theta)
        u_theta = data["u_theta"].values.reshape(N_t, N_r * N_theta)
        div_u = []
        for i in range(N_t):
            div_u_r = u_r[i] / r + kernels["psi"]["grad_r"].dot(u_r[i])
            div_u_theta = kernels["psi"]["grad_theta"].dot(u_theta[i])
            mag_u = np.sqrt(u_r[i]**2 + u_theta[i]**2)
            # div_u.append(self.beta*T_old[i]*(div_u_r + div_u_theta)*dt[i])
            div_u.append((div_u_r + div_u_theta)/mag_u)
        v = np.array(div_u).reshape(-1, X.shape[1], X.shape[0]-1).T
        print(v.min(), v.max(), v.mean(), np.nanmedian(np.abs(v)))
        v = np.concatenate([v, v[0, :, :].reshape(1, v.shape[1], len(t_vals))], axis=0)
        v0 = v[:, :, 0]

        Tmin, Tmax = T_range
        if Tmin is None:
            Tmin = np.min(T)
        if Tmax is None:
            Tmax = np.max(T)

        vmin, vmax = v_range
        if vmin is None:
            vmin = np.min(v)
        if vmax is None:
            vmax = np.max(v)

        # Model object perimeter
        circle = matplotlib.patches.Circle(
            xy=[0, 0], radius=(self.N_r-0.5)*self.dr,
            edgecolor="k", facecolor="none"
        )

        circle2 = matplotlib.patches.Circle(
            xy=[0, 0], radius=(self.N_r-1.5)*self.dr,
            edgecolor="k", facecolor="none"
        )

        # Prepare colour bar range/ticks/etc.
        cmap = "coolwarm"
        cbar_ticks_T = np.linspace(Tmin, Tmax, 11)
        sm_T = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=Tmin, vmax=Tmax)
        )
        sm_T._A = []

        cbar_ticks_v = np.linspace(vmin, vmax, 11)
        sm_v = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm_v._A = []

        self.fig = plt.figure(figsize=(12, 5))

        ax1 = self.fig.add_subplot(121, aspect="equal")
        ax1.add_patch(circle)
        plt.xticks([])
        plt.yticks([])
        plt.xlim((-1.1 * R, 1.1 * R))
        plt.ylim((-1.1 * R, 1.1 * R))
        plt.axis("off")
        plt.colorbar(sm_T, extend="both", ticks=cbar_ticks_T, label="temperature [K]")
        self.CS1 = plt.contourf(
            X, Y, T0, 200, cmap=cmap, vmin=Tmin, vmax=Tmax
        )
        self.timer = plt.text(-R, R, "")

        ax2 = self.fig.add_subplot(122, aspect="equal")
        ax2.add_patch(circle2)
        plt.xticks([])
        plt.yticks([])
        plt.xlim((-1.1 * R, 1.1 * R))
        plt.ylim((-1.1 * R, 1.1 * R))
        plt.axis("off")
        plt.colorbar(sm_v, extend="both", ticks=cbar_ticks_v, label=r"$\log_{10}$ velocity [m/s]")
        self.CS2 = plt.contourf(
            X, Y, v0, 200, cmap=cmap, vmin=vmin, vmax=vmax
        )
        plt.tight_layout()

        print("Rendering animation...")

        duration = 10.0
        self.ani_dict = {
            "tmax": duration,
            "X": X,
            "Y": Y,
            "T": T,
            "v": v,
            "cmap": cmap,
            "T_range": (Tmin, Tmax),
            "v_range": (vmin, vmax),
            "tmax_real": np.max(t_vals),
            "axes": (ax1, ax2),
        }
        ani = VideoClip(self.update_animation_save, duration=duration)
        ani.write_videofile("movie.mp4", fps=T.shape[0]/duration)

        # plt.show()
        pass


    def plot_frame(self, inds, T_range=(None, None), v_range=(None, None)):
        T = self.data["T"][inds].values
        T = T.reshape(self.N_r, self.N_theta).T
        v = np.log10(np.sqrt(self.data["u_r"]**2 + 0*self.data["u_theta"]**2)[inds]).values
        v = v.reshape(self.N_r, self.N_theta).T
        R = self.R

        circle = matplotlib.patches.Circle(
            xy=[0, 0], radius=(self.N_r-0.5)*self.dr,
            edgecolor="k", facecolor="none"
        )

        circle2 = matplotlib.patches.Circle(
            xy=[0, 0], radius=(self.N_r-2)*self.dr,
            edgecolor="k", facecolor="none"
        )

        # Map data for theta = 0 to theta = 2*pi
        # to make the contour plots periodic
        X = np.vstack([self.X, self.X[0]])
        Y = np.vstack([self.Y, self.Y[0]])
        T = np.vstack([T, T[0]])
        v = np.vstack([v, v[0]])

        Tmin, Tmax = T_range
        if Tmin is None:
            Tmin = np.nanmin(T)
        if Tmax is None:
            Tmax = np.nanmax(T)

        vmin, vmax = v_range
        if vmin is None:
            vmin = np.nanmin(v)
        if vmax is None:
            vmax = np.nanmax(v)

        # Prepare colour bar range/ticks/etc.
        cmap = "coolwarm"
        cbar_ticks_T = np.linspace(Tmin, Tmax, 11)
        sm_T = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=Tmin, vmax=Tmax)
        )
        sm_T._A = []

        cbar_ticks_v = np.linspace(vmin, vmax, 11)
        sm_v = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm_v._A = []

        plt.figure(figsize=(12, 5))
        ax = plt.subplot(121, aspect="equal")
        CS = plt.contourf(
            X, Y, T, 200, cmap=cmap, vmin=Tmin, vmax=Tmax
        )
        # plt.contour(X, Y, T, 10, cmap="Greys")
        ax.add_patch(circle)
        plt.xticks([])
        plt.yticks([])
        plt.xlim((-1.1 * R, 1.1 * R))
        plt.ylim((-1.1 * R, 1.1 * R))
        plt.axis("off")
        plt.colorbar(sm_T, extend="both", ticks=cbar_ticks_T, label="temperature [K]")

        ax2 = plt.subplot(122, aspect="equal")
        CS = plt.contourf(
            X, Y, v, 200, cmap=cmap, vmin=vmin, vmax=vmax
        )
        # plt.contour(X, Y, T, 10, cmap="Greys")
        ax2.add_patch(circle2)
        plt.xticks([])
        plt.yticks([])
        plt.xlim((-1.1 * R, 1.1 * R))
        plt.ylim((-1.1 * R, 1.1 * R))
        plt.axis("off")
        plt.colorbar(sm_v, extend="both", ticks=cbar_ticks_v, label=r"$\log_{10}$ velocity [m/s]")

        plt.tight_layout()
        plt.show()
        pass
