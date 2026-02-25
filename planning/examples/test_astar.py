"""
An example showing how to use A* on gridmap to plan a path and visualize it in mujoco.
We don't use the robot instance here, since we are planning on a gridmap instead of in mujoco sim
"""


import mujoco
import numpy as np
from m0.planning.a_star import graph_search
from m0.utils.gridmap_2d import GridMap
from m0.viewer.mujoco_visualization import MujocoViewer
import time




if __name__ == "__main__":

    # load mujoco model
    model = mujoco.MjModel.from_xml_path("m0/assets/test_world.xml")
    data = mujoco.MjData(model)

    mjv = MujocoViewer(model, data)

    # build 2D gridmap from mujoco model
    grid_map = GridMap(model=model, data=data, resolution=0.05, width=10.0, height=10.0,
                       robot_radius=0.3, margin=0.1)
    
    # plan a path using A*
    path = graph_search(start=[0, 0.2], goal=[9, 8.4], gridmap=grid_map)
    xyz_path = np.zeros((path.shape[0], 3)) + 0.03
    xyz_path[:, 0:2] = path

    # show in mujoco
    while mjv.is_running():
        mjv.draw_traj(xyz_path)
        mjv.render()
        time.sleep(0.5)

