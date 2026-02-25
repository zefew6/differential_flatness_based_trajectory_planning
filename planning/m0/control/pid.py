import numpy as np
from m0.utils.utils import wrap_pi


class PIDController:
    """
    NOTICE: Technically it's actually called: Exponential Stabilization of Unicycle
    control input: [v, w]
    """
    def __init__(self, 
                 goal,
                 max_v=1.0, 
                 max_w=1.0,
                 k_rho = 1.0,
                 k_alpha = 0.7,
                 k_beta = -0.5,
                 goal_tol=0.05,
                 use_speed_gate=True,
                 can_backword=False):
        
        """
        goal: [x,y,yaw]
        max_v: max linear velocity
        max_w: max angular velocity
        k*: PID params
        goal_tol: how close to the goal can be seen as reached goal
        use_speed_gate: if use speed gate
        can_backword: if go backward can be allowed
        """
        
        self.goal = goal
        self.max_v = max_v
        self.max_w = max_w
        self.goal_tol = goal_tol
        self.k_rho = k_rho      # dist convergence speed
        self.k_alpha = k_alpha   # heading convergence speed
        self.k_beta = k_beta     # final orientation adjustment, must be negative

        self.use_speed_gate = use_speed_gate

        if not can_backword:
            self.v_limits = np.array([0, max_v])
        else:
            self.v_limits = np.array([-max_v, max_v])
        self.w_limits = np.array([-max_w, max_w])


        self.reached = False



    def set_goal(self, goal):
        self.goal = goal
        self.reached = False



    def step(self, state):

        """
        state: [x,y,yaw,vx,vy,yaw_dot]
        """

        x, y, yaw = state[:3]
        goal_x, goal_y, goal_yaw = self.goal[0], self.goal[1], self.goal[2]

        dx, dy = goal_x - x, goal_y - y
        rho = np.hypot(dx, dy)
        alpha = wrap_pi(np.arctan2(dy, dx) - yaw)
        beta  = wrap_pi(goal_yaw - yaw - alpha)

        v = self.k_rho * rho
        w = self.k_alpha * alpha + self.k_beta * beta

        # limit
        v = np.clip(v, 0, self.max_v)
        w = np.clip(w, -self.max_w, self.max_w)


        if rho < self.goal_tol:
            v = 0
            if abs(wrap_pi(goal_yaw - yaw)) < 0.05:
                w = 0
            else:
                w = self.k_beta * wrap_pi(goal_yaw - yaw)
        return v, w