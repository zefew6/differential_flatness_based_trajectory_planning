"""
Example script showing how to use PID to control a differential robot
"""

import mujoco
import time
from m0.viewer.mujoco_visualization import MujocoViewer
from m0.control.pid import PIDController
from m0.robot.robot import Robot

import numpy as np



if __name__ == "__main__":

    # init model
    model = mujoco.MjModel.from_xml_path("m0/assets/rl_scene.xml")
    data = mujoco.MjData(model)

    # init viewer
    mjv = MujocoViewer(model, data)

    # create robot instance
    robot1 = Robot(model, data, robot_body_name="pusher1", actuator_names=["forward", "turn"], max_v=1.0, max_w=1.0)

    # goal: 
    goal = np.array([-2.0, 3.0, -np.pi])

    controller = PIDController(goal, 
                               max_v=robot1.max_v, 
                               max_w=robot1.max_w)
    
    # draw the goal
    mjv.draw_point(np.array([goal[0], goal[1], 0.03]))

    # frame_skip(control rate): we don't want to control it every time
    frame_skip = 5

    # main loop
    while mjv.is_running():
        
        # get the robot's current state
        pos_yaw = robot1.get_state() # [x, y, yaw]
        vel6 = robot1.get_v() #[vx, vy, vz, wx, wy, wz]
        lin_v, ang_w = vel6[:3], vel6[3:]
        state = np.array([pos_yaw[0], pos_yaw[1], pos_yaw[2],
                          lin_v[0], lin_v[1], ang_w[2]])


        # controller output
        v, omega = controller.step(state)

        # send ctrl to robot
        robot1.set_ctrl(v, omega)

        # simulate frame_skip step
        for _ in range(frame_skip):
            mujoco.mj_step(model, data)

        # mujoco show
        mjv.render()

        # slow down
        time.sleep(0.001)

    # stop listening
    mjv.close()