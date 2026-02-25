"""
using ompl control-based planner to plan a kinodynamic path for robot
it's now kinodynamic feasible, but it can't actually reach the goal, while can reach the nearby area.
"""

from ompl import base as ob
from ompl import control as oc
import math
import mujoco
import mujoco.viewer
import numpy as np
import time


class KinodynamicRRT:
    def __init__(self, robot, start, goal):
        """
        robot: Robot instance
        start: [x, y, theta]
        goal: [x, y, theta]
        """

        self.robot = robot
        self.start = start
        self.goal = goal

        # state space here (yaw has been already considered in SE2)
        self.space = ob.SE2StateSpace()
        self.bounds = ob.RealVectorBounds(2)
        self.bounds.setLow(-5)
        self.bounds.setHigh(5)
        self.space.setBounds(self.bounds)


        ## control space here
        self.cspace = oc.RealVectorControlSpace(self.space, 2) # v, w
        self.c_bounds = ob.RealVectorBounds(2)
        self.c_bounds.setLow(0, 0.0)   # v ∈ [0, 1] we don't allow backward movement
        self.c_bounds.setHigh(0,  1.0)
        self.c_bounds.setLow(1, -1.0)   # w ∈ [-1, 1]
        self.c_bounds.setHigh(1,  1.0)
        self.cspace.setBounds(self.c_bounds)


    # check collision function
    def is_state_valid(self, state):
        se2 = state
        x = se2.getX()
        y = se2.getY()
        theta = se2.getYaw()

        z = 0.30  # HAVE TO SET HIGHER Z VALUE TO AVOID COLLISION WITH THE GROUND
        self.robot.set_state([x, y, z], theta)
        mujoco.mj_forward(self.robot.model, self.robot.data)

        return self.robot.is_collid() == False  


    def propagate(self, start, control, duration, result):
        # consider the solving speed, using simple euler integration
        x = start.getX()
        y = start.getY()
        theta = start.getYaw()

        v = control[0]
        w = control[1]

        dt = self.si.getPropagationStepSize()
        steps = max(1, int(duration / dt))
        rem = duration - steps * dt

        for _ in range(steps):
            state = self.robot.discrete_dyn([x, y, theta], [v, w], dt)
            x, y, theta = state[0], state[1], state[2]
        if rem > 1e-9:
            state = self.robot.discrete_dyn([x, y, theta], [v, w], rem)
            x, y, theta = state[0], state[1], state[2]

        result.setX(x)
        result.setY(y)
        result.setYaw(theta)


    def solve(self):
        # SimpleSetup
        self.ss = oc.SimpleSetup(self.cspace)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))

        self.si = self.ss.getSpaceInformation()
        self.si.setPropagationStepSize(0.005)             # dt, same as mujoco timestep, but slow, can be a little larger
        self.si.setMinMaxControlDuration(1, 12)         # 1*dt ~ 20*dt, randomly sampled
        self.si.setStateValidityCheckingResolution(0.02) # check sampling resolution

        # start and goal
        start = ob.State(self.space)
        start().setX(self.start[0]); start().setY(self.start[1]); start().setYaw(self.start[2])
        goal = ob.State(self.space)
        goal().setX(self.goal[0]); goal().setY(self.goal[1]); goal().setYaw(self.goal[2])
        self.ss.setStartAndGoalStates(start, goal, 0.1)

        # Planner: RRT
        self.planner = oc.RRT(self.si)
        self.ss.setPlanner(self.planner)


        self.final_path = []
        if self.ss.solve(20.0):
            path = self.ss.getSolutionPath()
            print("Found solution:")
            for i in range(path.getStateCount()):
                state = path.getState(i)
                self.final_path.append([state.getX(), state.getY(), state.getYaw()])

        return self.final_path

