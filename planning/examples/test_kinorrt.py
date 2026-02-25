import numpy as np
import mujoco
import time
from m0.planning.kino_rrt import KinodynamicRRT
from m0.robot.robot import Robot
from m0.viewer.mujoco_visualization import MujocoViewer




if __name__ == "__main__":

    # load mujoco model
    model = mujoco.MjModel.from_xml_path("m0/assets/rrt_connect_scene.xml")
    data = mujoco.MjData(model)

    # create robot instance
    robot = Robot(model, data, robot_body_name="pusher1", actuator_names=["forward", "turn"], max_v=1.0, max_w=1.0)

    # viewer init
    mjv = MujocoViewer(model, data)

    # planner
    start = [-4, -4, 0.0]
    goal = [3.6, 0, np.pi]
    start_xyz = np.array([start[0], start[1], 0.3])
    goal_xyz = np.array([goal[0], goal[1], 0.3])

    kino_rrt = KinodynamicRRT(robot, start, goal)
    final_path = kino_rrt.solve()
    final_path_xyz = np.zeros((len(final_path), 3)) + 0.3
    final_path_xyz[:, 0:2] = np.array(final_path)[:, 0:2]

    # draw in mujoco
    mjv.draw_point(start_xyz, size=0.05, rgba=np.array([1, 0, 0, 1]))
    mjv.draw_point(goal_xyz, size=0.05, rgba=np.array([0, 1, 0, 1]))
    mjv.draw_traj(final_path_xyz, size=0.02, rgba=np.array([0.1, 0.5, 1, 1]))


    # loop to follow the path
    while mjv.is_running():
        for state in final_path:
            # set robot state
            robot.set_state([state[0], state[1], 0.03], state[2])
            mujoco.mj_forward(model, data)
            # render
            mjv.render()
            # slow down
            time.sleep(0.02)
