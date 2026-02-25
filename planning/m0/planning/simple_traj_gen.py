import numpy as np

class SimpleTrajectoryGenerator:
    """
    General 2D trajectory generator with time parameterization.
    Supports: line, circle, ellipse, lemniscate (∞), s_curve, figure8.
    Provides position, velocity, acceleration, jerk, snap, and yaw.
    """

    def __init__(self, traj_type="circle", center=(0., 0.), params=None,
                 duration=10.0, yaw_mode="tangent", phase=0.0, repeat=True):
        """
        Args:
            traj_type: str, one of ['line', 'circle', 'ellipse', 'lemniscate', 's_curve', 'figure8']
            center   : (x,y), trajectory center in world frame
            params   : dict, type-specific parameters
            duration : float, total period (s)
            yaw_mode : 'tangent' (heading follows direction)
                        or 'sin' (yaw oscillates)
            phase    : float, phase offset in seconds
            repeat   : bool, if False then time is clamped at duration
                       (trajectory stops at final point)
        """
        self.traj_type = traj_type
        self.center = np.array(center, dtype=float)
        self.params = params or {}
        self.duration = duration
        self.phase = phase
        self.repeat = repeat
        self.omega = 2 * np.pi / duration
        self.yaw_mode = yaw_mode

    # ------------------------------------------------
    def update(self, t):
        """Compute trajectory state at time t."""
        # --- ✅ 1️⃣ Clamp or wrap time ---
        if not self.repeat:
            # Clamp to maximum duration
            t = min(t, self.duration)
        else:
            # Wrap around for periodic trajectories
            t = t % self.duration

        tau = self.omega * (t + self.phase)

        # --- 2️⃣ Generate trajectory by type ---
        if self.traj_type == "circle":
            r = self.params.get("radius", 1.0)
            x = self.center + np.array([r*np.cos(tau), r*np.sin(tau)])
            x_dot = np.array([-r*self.omega*np.sin(tau), r*self.omega*np.cos(tau)])
            x_ddot = np.array([-r*(self.omega**2)*np.cos(tau), -r*(self.omega**2)*np.sin(tau)])
            x_dddot = np.array([r*(self.omega**3)*np.sin(tau), -r*(self.omega**3)*np.cos(tau)])
            x_ddddot = np.array([r*(self.omega**4)*np.cos(tau), r*(self.omega**4)*np.sin(tau)])

        elif self.traj_type == "ellipse":
            a = self.params.get("a", 1.0)
            b = self.params.get("b", 0.5)
            x = self.center + np.array([a*np.cos(tau), b*np.sin(tau)])
            x_dot = np.array([-a*self.omega*np.sin(tau), b*self.omega*np.cos(tau)])
            x_ddot = np.array([-a*(self.omega**2)*np.cos(tau), -b*(self.omega**2)*np.sin(tau)])
            x_dddot = np.array([a*(self.omega**3)*np.sin(tau), -b*(self.omega**3)*np.cos(tau)])
            x_ddddot = np.array([a*(self.omega**4)*np.cos(tau), b*(self.omega**4)*np.sin(tau)])

        elif self.traj_type == "line":
            start = np.array(self.params.get("start", [-1.0, 0.0]))
            end = np.array(self.params.get("end", [1.0, 0.0]))
            T = self.duration
            alpha = np.clip(t / T, 0, 1)
            x = (1 - alpha) * start + alpha * end
            x_dot = (end - start) / T
            x_ddot = x_dddot = x_ddddot = np.zeros(2)

        elif self.traj_type == "lemniscate":
            a = self.params.get("a", 1.0)
            x = self.center + np.array([
                a * np.cos(tau),
                a * np.sin(tau) * np.cos(tau)
            ])
            x_dot = np.array([
                -a*self.omega*np.sin(tau),
                a*self.omega*np.cos(2*tau)
            ])
            x_ddot = np.array([
                -a*(self.omega**2)*np.cos(tau),
                -2*a*(self.omega**2)*np.sin(2*tau)
            ])
            x_dddot = np.array([
                a*(self.omega**3)*np.sin(tau),
                -4*a*(self.omega**3)*np.cos(2*tau)
            ])
            x_ddddot = np.array([
                a*(self.omega**4)*np.cos(tau),
                8*a*(self.omega**4)*np.sin(2*tau)
            ])

        elif self.traj_type == "figure8":
            a = self.params.get("a", 1.0)
            x = self.center + np.array([
                a * np.sin(tau),
                0.5 * a * np.sin(2 * tau)
            ])
            x_dot = np.array([
                a * self.omega * np.cos(tau),
                a * self.omega * np.cos(2 * tau)
            ])
            x_ddot = np.array([
                -a * (self.omega**2) * np.sin(tau),
                -2 * a * (self.omega**2) * np.sin(2 * tau)
            ])
            x_dddot = np.array([
                -a * (self.omega**3) * np.cos(tau),
                -4 * a * (self.omega**3) * np.cos(2 * tau)
            ])
            x_ddddot = np.array([
                a * (self.omega**4) * np.sin(tau),
                8 * a * (self.omega**4) * np.sin(2 * tau)
            ])

        elif self.traj_type == "s_curve":
            L = self.params.get("length", 4.0)
            H = self.params.get("height", 1.0)
            x = self.center + np.array([
                L * (tau / (2*np.pi) - 0.5),
                H * np.sin(tau)
            ])
            x_dot = np.array([
                L * self.omega / (2*np.pi),
                H * self.omega * np.cos(tau)
            ])
            x_ddot = np.array([0, -H*(self.omega**2)*np.sin(tau)])
            x_dddot = np.array([0, -H*(self.omega**3)*np.cos(tau)])
            x_ddddot = np.array([0, H*(self.omega**4)*np.sin(tau)])

        else:
            raise ValueError(f"Unknown trajectory type '{self.traj_type}'")

        # --- 3️⃣ heading (yaw) ---
        if self.yaw_mode == "tangent":
            theta = np.arctan2(x_dot[1], x_dot[0])
        elif self.yaw_mode == "sin":
            theta = np.sin(self.omega * t)
        else:
            theta = 0.0

        return dict(
            x=x,
            x_dot=x_dot,
            x_ddot=x_ddot,
            x_dddot=x_dddot,
            x_ddddot=x_ddddot,
            theta=theta,
            t=t
        )


# ------------------ test -------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    traj = SimpleTrajectoryGenerator("figure8", params={"a": 1.0}, duration=5.0, repeat=False)
    ts = np.linspace(0, 10, 400)
    pts = np.array([traj.update(t)["x"] for t in ts])

    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1], "r")
    plt.axis("equal")
    plt.title("Figure-8 Trajectory with Duration Clamping")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.show()
