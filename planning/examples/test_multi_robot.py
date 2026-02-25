"""
Example script showing how to manage a multi-robots environment using m0
"""

import mujoco
import time
import numpy as np
from m0.robot.robot import Robot
from m0.viewer.mujoco_visualization import MujocoViewer


if __name__ == "__main__":

    # load xml
    model = mujoco.MjModel.from_xml_path("m0/assets/scene_p170_r5.xml")
    data = mujoco.MjData(model)

    # init viewer
    mjv = MujocoViewer(model, data)

    # create multi-robot instances
    robots = []
    for i in range(5):
        body_name = f"{i}" # from xml file
        robots.append(Robot(model, data, body_name, actuator_names=["forward_"+str(i), "turn_"+str(i)], max_v=1.0, max_w=1.0))

    # main loop
    for step in range(200000):
        for i, r in enumerate(robots):
            v = 0.5
            w = 0.5 * np.sin(0.01 * step + i)
            r.set_ctrl(v, w)
        mujoco.mj_step(model, data)

        mjv.render()
        time.sleep(0.0001)

    mjv.close()
