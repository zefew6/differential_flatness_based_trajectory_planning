"""
Example script showing how to use mpc to control a differential robot follow a time-parametrized path
"""

import mujoco
import time
import numpy as np
from m0.robot.robot import Robot
from m0.viewer.mujoco_visualization import MujocoViewer
from m0.control.mpc_acados import MPCController
from m0.utils.utils import wrap_pi
from m0.planning.simple_traj_gen import SimpleTrajectoryGenerator


if __name__ == "__main__":

    # load xml
    model = mujoco.MjModel.from_xml_path("m0/assets/rl_scene.xml")
    data = mujoco.MjData(model)


    # init viewer
    mjv = MujocoViewer(model, data)
    mjv.set_camera(distance=10.0, azimuth=90, elevation=-30, lookat=[0, 0, 0])
    # create robot instance in sim
    robot1 = Robot(model, data, robot_body_name="pusher1", actuator_names=["forward", "turn"], max_v=1.0, max_w=1.0)

    # get the initial state(make sure it's valid) and control input of robot
    x0 = robot1.get_state()
    u0 = robot1.get_ctrl()

    # init controller
    controller = MPCController(x0, u0)

    traj = SimpleTrajectoryGenerator("figure8", params={"a": 3.0}, duration=60, repeat=True)


    # sample point and draw
    
    ts = np.linspace(0, traj.duration, 1000)
    traj_points = np.array([traj.update(t)["x"] for t in ts])
    traj_points = np.hstack((traj_points, 0.03 * np.ones((traj_points.shape[0], 1))))
    mjv.draw_traj(traj_points)


    # current sim time
    t = data.time
    step = 0

    # get the ctrl frequency, Since the dt(50ms default) of dynamics in mpc
    # does not match the mujoco simulation timestep(2ms default), 
    # for better matching, we apply the same control input for k_step times in sim
    k_step = int(round(controller.dt / model.opt.timestep))

    # loop
    y_ref = np.zeros((controller.horizon,5))  # x, y, theta, v, w
    y_refN = np.zeros((5,))      # x, y, theta, v, w

    while mjv.is_running():

        # get current y_ref
        for j in range(controller.horizon):
            tj = t + j * controller.dt
            traj_state = traj.update(tj)
            xr, yr = traj_state["x"]
            thr = traj_state["theta"]
            # angle wrapper
            thr_wrapped = x0[2] + wrap_pi(thr - x0[2])
            y_ref[j] = [xr, yr, thr_wrapped, 0, 0]

        traj_state = traj.update(t + controller.horizon * controller.dt)
        xN, yN  = traj_state["x"]
        thN = traj_state["theta"]
        thN_wrapped = x0[2] + wrap_pi(thN - x0[2])
        # only need state at the end of horizon
        y_refN = np.array([xN, yN, thN_wrapped])


        # mpc step
        u0 = controller.step(x0, y_ref, y_refN)

        # send ctrl to robot to k_step times
        for _ in range(k_step):
            robot1.set_ctrl(u0[0], u0[1])
            mujoco.mj_step(model, data)
        
        # update time
        t = data.time

        # update current state of robot
        x0 = robot1.get_state()


        # print info
        iters = controller.solver.get_stats('sqp_iter')
        cost  = controller.solver.get_cost()
        if controller.status not in [0, 2]:
            print(f"step={step} t={t:.2f} pos=({x0[0]:.3f},{x0[1]:.3f}) yaw={x0[2]:.2f} "
                f"u={u0} status={controller.status} iters={iters} cost={cost:.3f}")

        # mujoco show
        mjv.render()

        # slow down
        time.sleep(0.01)

        step += 1
