"""
Example script showing how to use keyboard to control a differential robot
"""


import mujoco
import time
from m0.viewer.mujoco_visualization import MujocoViewer
from m0.control.keyboard import KeyboardController
from m0.robot.robot import Robot



if __name__ == "__main__":

    # init model
    model = mujoco.MjModel.from_xml_path("m0/assets/rl_scene.xml")
    data = mujoco.MjData(model)

    # init viewer
    mjv = MujocoViewer(model, data)

    # create robot instance
    robot1 = Robot(model, data, robot_body_name="pusher1", actuator_names=["forward", "turn"], max_v=1.0, max_w=1.0)

    controller = KeyboardController(max_v=robot1.max_v, max_w=robot1.max_w)

    # main loop
    while mjv.is_running():

        # controller output
        v, omega = controller.step()

        # send ctrl to robot
        robot1.set_ctrl(v, omega)

        # simulate one step
        mujoco.mj_step(model, data)

        # mujoco show
        mjv.render()

        # slow down
        time.sleep(0.001)

    # stop listening
    controller.stop()
    mjv.close()