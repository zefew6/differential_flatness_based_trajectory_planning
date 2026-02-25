from acados_template.acados_sim_solver import AcadosSimSolver
import numpy as np
import casadi
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, acados_ocp_solver
import scipy


class MPCController:
    def __init__(self,
                 X0,
                 U0,
                 horizon=12, 
                 dt=0.05, 
                 Q = np.diag([20, 20, 2]),
                 R = np.diag([0.1, 0.1]),
                 max_v = 1.0,
                 max_w = 1.0,
                 can_backward = False,
                 solver_name="acados_solver"):
        """
        X0: the initial state of robot [x, y, theta]
        U0: the initial control input of robot [v,w]
        horizon: control horizon of mpc (steps)
        dt: time step of discretization of nonlinear dynamics
        Q, R: cost parameters in objective function
        max_v, max_w: max linear and angular velocity
        can_backward: if the car can go backward
        """
        self.horizon = horizon
        self.dt = dt
        self.solver_name = solver_name
        self.Q = Q
        self.R = R
        self.max_v = max_v
        self.max_w = max_w
        self.can_backward = can_backward

        # record mpc solution last time for warm-start
        self.u_prev = np.zeros((self.horizon, U0.shape[0]))
        self.x_prev = np.tile(X0, (self.horizon+1, 1))

        self.ocp = self._create_ocp()
        self.solver = AcadosOcpSolver(self.ocp)
        self.integrator = AcadosSimSolver(self.ocp)

    # ---------------- Diff-drive dynamics ----------------
    def dynamics(self):
        x = casadi.SX.sym("x")
        y = casadi.SX.sym("y")
        th = casadi.SX.sym("th")
        states = casadi.vertcat(x, y, th)

        v = casadi.SX.sym("v")
        w = casadi.SX.sym("w")
        controls = casadi.vertcat(v, w)

        x_dot = casadi.SX.sym("x_dot")
        y_dot = casadi.SX.sym("y_dot")
        theta_dot = casadi.SX.sym("theta_dot")
        states_dot = casadi.vertcat(x_dot, y_dot, theta_dot)

        f_expl = casadi.vertcat(v * casadi.cos(th), v * casadi.sin(th), w)

        model = AcadosModel()
        model.name = "diffdrive"
        model.x = states
        model.xdot = states_dot
        model.u = controls
        model.f_expl_expr = f_expl
        model.f_impl_expr = states_dot - f_expl
        return model

    # ---------------- Build OCP ----------------
    def _create_ocp(self):
        ocp = AcadosOcp()
        self.model = self.dynamics()
        ocp.model = self.model

        nx = self.model.x.rows()
        nu = self.model.u.rows()

        ocp.solver_options.N_horizon = self.horizon

        Q = self.Q
        R = self.R

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ocp.cost.W_e = Q
        ocp.cost.W = scipy.linalg.block_diag(Q, R)

        ocp.cost.Vx = np.zeros((nx+nu, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        Vu = np.zeros((nx+nu, nu))
        Vu[nx:nx+nu, :nu] = np.eye(nu)
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros(nx + nu)
        ocp.cost.yref_e = np.zeros(nx)

        max_v, max_w = self.max_v, self.max_w
        if not self.can_backward:
            ocp.constraints.lbu = np.array([0., -max_w])
        else:
            ocp.constraints.lbu = np.array([-max_v, -max_w])
        ocp.constraints.ubu = np.array([+max_v, +max_w])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = np.zeros(nx)

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.tf = self.horizon * self.dt
        return ocp
    

    def step(self, x0, y_ref, y_refN):
        """
        x0: the current true state: [x,y,yaw]; size=(3,1)
        y_ref: the reference state: [x,y,theta,v_ref,w_ref]; size=(N_horizon, 5)
        y_refN: the reference state at horizon N: [x,y,theta,v_ref,w_ref]; size=(5, 1)
        """
        # set the N_horizon reference:
        for i in range(self.horizon):
            yref_j = y_ref[i].T.copy()
            self.solver.set(i, "yref", yref_j)

        # set the terminal reference:
        self.solver.set(self.horizon, "yref", y_refN)

        # set x = x0
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # warm_start
        for i in range(self.horizon):
            self.solver.set(i, "x", self.x_prev[i+1])
            self.solver.set(i, "u", self.u_prev[i])
        self.solver.set(self.horizon, "x", self.x_prev[-1])

        # solve
        status = self.solver.solve()
        if status not in (0, 2): # 0: success, 2: max iter reach
            # second chance with looser setting
            self.solver.options_set('levenberg_marquardt', 1.0e-2)
            status = self.solver.solve()

        if status in (0, 2):
            u0 = self.solver.get(0, "u")

            # record the solution this time for next time warmstart
            for j in range(self.horizon):
                self.u_prev[j] = self.solver.get(j, "u")
                self.x_prev[j] = self.solver.get(j, "x")
            self.x_prev[self.horizon] = self.solver.get(self.horizon, "x")

        # if fail, stick to the last solution
        else:
            u0 = self.u_prev[0].copy()
            print(f"[MPC]: acados fail to solve, status={status}, keep last u.")

        self.status = status


        return u0[0], u0[1]
    
