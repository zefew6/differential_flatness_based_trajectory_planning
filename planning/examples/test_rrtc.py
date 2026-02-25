"""
An example showing how to use RRT-Connect to plan a path and visualize it in mujoco.
"""

from m0.planning.rrt_connect import RRTConnect
from m0.viewer.mujoco_visualization import MujocoViewer
from m0.robot.robot import Robot
import mujoco
import numpy as np
import time




if __name__ == "__main__":

    # load mujoco model
    model = mujoco.MjModel.from_xml_path("m0/assets/rrt_connect_scene.xml")
    data = mujoco.MjData(model)

    robot = Robot(model, data, robot_body_name="pusher1", actuator_names=["forward", "turn"], max_v=1.0, max_w=1.0)

    # plan a path using RRT-Connect
    start = [-2.0, 2.0, 0.0]
    goal = [3.8, -0.0, np.pi/2]
    rrt = RRTConnect(robot, start, goal)
    
    # viewer
    mjv = MujocoViewer(model, data)
    
    solved, path = rrt.solve()
    print("Solved:", solved)


    # visualize the path in mujoco
    start_xyz = np.array([start[0], start[1], 0.3])
    goal_xyz = np.array([goal[0], goal[1], 0.3])
    path_xyz = np.zeros((len(path), 3)) + 0.3   
    path_xyz[:, 0:2] = np.array(path)[:, 0:2]

    mjv.draw_point(start_xyz, size=0.05, rgba=np.array([1, 0, 0, 1]))
    mjv.draw_point(goal_xyz, size=0.05, rgba=np.array([0, 1, 0, 1]))
    mjv.draw_traj(path_xyz, size=0.02, rgba=np.array([0.1, 0.5, 1, 1]))

    
    # draw robot along the path in mujoco
    i = 0
    while mjv.is_running():
        if i < len(path):
            robot.set_state([path[i][0], path[i][1], 0.03], path[i][2])
            mujoco.mj_forward(model, data)
            i += 1

        time.sleep(0.01)
        mjv.render()