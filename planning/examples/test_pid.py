"""
Example script showing how to use PID to control a differential robot
"""

import mujoco
import time
import numpy as np
from m0.viewer.mujoco_visualization import MujocoViewer
from m0.control.pid import PIDController
from m0.robot.robot import Robot

def sample_random_goal(xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0):
    return np.array([
        np.random.uniform(xmin, xmax),
        np.random.uniform(ymin, ymax),
        np.random.uniform(-np.pi, np.pi),
    ], dtype=float)

if __name__ == "__main__":
    # init model and data
    model = mujoco.MjModel.from_xml_path("m0/assets/rl_scene.xml")
    data = mujoco.MjData(model)

    # init viewer
    mjv = MujocoViewer(model, data)
    # 设置相机视角: distance, azimuth, elevation, lookat
    mjv.set_camera(distance=10.0, azimuth=90, elevation=-30, lookat=[0, 0, 0])

    # create robot instance
    robot1 = Robot(model, data, robot_body_name="pusher1", 
                   actuator_names=["forward", "turn"], max_v=1.0, max_w=1.0)

    # initialize goal
    goal = sample_random_goal()
    controller = PIDController(goal, max_v=robot1.max_v, max_w=robot1.max_w)
    mjv.draw_point(np.array([goal[0], goal[1], 0.03]))

    frame_skip = 5  # control rate

    # main loop
    while mjv.is_running():
        # get robot's state: [x, y, yaw, vx, vy, w]
        pos_yaw = robot1.get_state()
        vel6 = robot1.get_v()
        state = np.array([pos_yaw[0], pos_yaw[1], pos_yaw[2],
                          vel6[0], vel6[1], vel6[5]])

        # check if goal reached
        if np.hypot(pos_yaw[0]-goal[0], pos_yaw[1]-goal[1]) <= controller.goal_tol:
            mjv.reset(0)
            # generate random goal
            goal = sample_random_goal()
            controller.set_goal(goal)
            mjv.draw_point(np.array([goal[0], goal[1], 0.03]))
            print(f"New goal: x={goal[0]:.2f}, y={goal[1]:.2f}, yaw={goal[2]:.2f}")
        
        # compute control
        v, omega = controller.step(state)
        robot1.set_ctrl(v, omega)

        # simulate
        for _ in range(frame_skip):
            mujoco.mj_step(model, data)

        mjv.render()
        time.sleep(0.001)

    mjv.close()