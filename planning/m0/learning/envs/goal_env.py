import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

from m0.utils.utils import wrap_pi
from m0.viewer.mujoco_visualization import MujocoViewer
from m0.robot.robot import Robot
import mujoco


class DiffDrivePointGoalEnv(gym.Env):
    """
    Point to point navigation for differential drive robot
    Observation: 6-dim: [dx, dy, dist, angle_err, prev_v, prev_w]
    Action: 2-dim: [v, w] (linear and angular velocities)
    Termination conditions: reach goal or max duration reached or out of bounds
    """

    def __init__(
        self,
        xml_path="m0/assets/rl_scene.xml",
        frame_skip=5,
        max_duration=10,
        seed=0,
        render=False,
    ):
        
        """
        xml_path: path to mujoco xml file
        frame_skip: number of sim steps for each action
        max_duration: max duration of one episode (seconds)
        seed: random seed
        render: if render the sim
        """

        super().__init__()

        # load mujoco model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # create robot instance
        self.robot = Robot(self.model, self.data, 
                      robot_body_name="pusher1",
                      actuator_names=["forward", "turn"], 
                      max_v=1.0, max_w=1.0)

        self.frame_skip = int(frame_skip)
        self.duration = float(max_duration)  # in seconds

        

        # action: [v, w]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # observation: [dx, dy, dist, angle_err, prev_v, prev_w]
        high = np.array([np.inf] * 6, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, shape=(6,), dtype=np.float32)

        # previous state
        self.prev_u = np.zeros(2, dtype=np.float64)
        self.prev_dist = 0.0
        self.goal = np.array([0.0, 0.0], dtype=np.float64)

        # seed
        self.seed = seed

        # option
        self.option = 'random'
        self.start_x_range = (-3.0, 3.0)
        self.start_y_range = (-3.0, 3.0)
        self.start_yaw_range = (-np.pi, np.pi)
        self.goal_x_range = (-4.0, 4.0)
        self.goal_y_range = (-4.0, 4.0)
        self.min_start_goal_dist = 2.5
        self.goal_tol = 0.2

        self.viewer = None
        if render:
            self.viewer = MujocoViewer(self.model, self.data)

        self.step_count = 0

    # ---- util ----
    def _get_pose(self):
        return self.robot.get_pos()[:2], self.robot.get_yaw()

    def _set_pose(self, xy, yaw, z=0.03):
        xyz = np.array([xy[0], xy[1], z], dtype=np.float64)
        self.robot.set_state(xyz, yaw)

    def _get_obs(self):
        pos_xy, yaw = self._get_pose()
        d = self.goal - pos_xy
        dist = float(np.linalg.norm(d))

        bearing = np.arctan2(d[1], d[0]) if dist > 1e-8 else yaw
        bearing_err = wrap_pi(bearing - yaw)
        obs = np.array([d[0], d[1], dist, bearing_err, self.prev_u[0], self.prev_u[1]], dtype=np.float32)
        return obs, dist, bearing_err



    # ---- template API ----
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        if self.viewer is not None:
            self.viewer.reset(0)

        mujoco.mj_resetData(self.model, self.data)


        if self.option == 'random':
            sx = float(self.np_random.uniform(*self.start_x_range))
            sy = float(self.np_random.uniform(*self.start_y_range))
            syaw = float(self.np_random.uniform(*self.start_yaw_range))
            start_xy = np.array([sx, sy], dtype=np.float64)
            start_yaw = syaw

            # ---------- random goal generation ----------
            while True:
                gx = float(self.np_random.uniform(*self.goal_x_range))
                gy = float(self.np_random.uniform(*self.goal_y_range))
                self.goal = np.array([gx, gy], dtype=np.float64)
                if np.linalg.norm(self.goal - start_xy) >= self.min_start_goal_dist:
                    break

        
        # draw the goal position in mujoco viewer
        if self.viewer is not None:
            pt = np.array([self.goal[0], self.goal[1], 0.03])
            self.viewer.draw_point(pt)


        self._set_pose(start_xy, start_yaw)
        mujoco.mj_forward(self.model, self.data)

        self.prev_u = np.zeros(2, dtype=np.float64)

        obs, dist, _ = self._get_obs()
        self.prev_dist = dist
        info = {}

        return obs, info

    def step(self, action):
        # action clip
        a = np.clip(action, -1.0, 1.0).astype(np.float64)
        v = a[0]
        w = a[1]

        # send to mujoco sim
        for _ in range(self.frame_skip):
            self.robot.set_ctrl(v, w)
            mujoco.mj_step(self.model, self.data)

        # observation & error
        obs, dist, bearing_err = self._get_obs()
        

        # reward：
        # 1) get reward by progress
        # 2) get reward by proximity
        # 3) small penalty: angle error, control effort & jitter
        progress = float(self.prev_dist - dist)
        shaping = float(np.exp(-1.2 * dist))
        control_penalty = 0.005 * float(v * v + w * w)
        smooth_penalty = 0.01 * float(np.sum((a - self.prev_u) ** 2))
        angle_penalty = 0.02 * float(abs(bearing_err))
        time_penalty = 0.05

        reward = 1.5 * progress + 1.1 * shaping - control_penalty - smooth_penalty - angle_penalty - time_penalty

        self.prev_u = a.copy()
        self.prev_dist = dist
        self.step_count += 1

        # termination
        terminated = False
        truncated = False
        info = {}

        pos = self.robot.get_pos()[:2]

        # success:
        # abs(v) <= 0.02 and abs(w) <= 0.02 will make the car jitter around the goal
        # simple way is to set the control to zero when reaching the goal
        if dist < self.goal_tol:
            reward += 200.0
            terminated = True
            info["success"] = True
            print("Goal reached! Distance to goal:", dist)

        # reach limit
        elif self.data.time >= self.duration:
            truncated = True
            print("Step limit reached. Distance to goal:", dist)

        # out of bounds
        elif pos[0] < -5 or pos[0] > 5 or pos[1] < -5 or pos[1] > 5:
            reward += -100.0
            terminated = True
            info["success"] = False
            print("Out of bounds! Pos:", pos)

        if self.viewer is not None:
            self.render()

        return obs, reward, terminated, truncated, info
    

    def render(self):
        if self.viewer.is_running():
            self.viewer.render()