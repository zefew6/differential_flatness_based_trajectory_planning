"""
RRT_CONNECT in OMPL to plan a collision-free path, pure geometric planning.
"""

import mujoco
import ompl.base as ob
import ompl.geometric as og
import numpy as np

from m0.robot.robot import Robot


class RRTConnect():
    def __init__(self, robot, start, goal):
        """
        robot: Robot instance
        start: [x, y, theta]
        goal: [x, y, theta]
        """

        self.robot = robot

        # set up ompl
        state_space = ob.RealVectorStateSpace(3)  # x, y, theta
        bounds = ob.RealVectorBounds(3)  # x, y, theta
        # for x:
        bounds.setLow(0, -5.0)
        bounds.setHigh(0, 5.0)
        # for y:
        bounds.setLow(1, -5.0)
        bounds.setHigh(1, 5.0)
        # for theta:
        bounds.setLow(2, -np.pi)
        bounds.setHigh(2, np.pi)


        state_space.setBounds(bounds)
        self.si = ob.SpaceInformation(state_space)


        def is_state_valid(state):  
            x = state[0]
            y = state[1]
            z = 0.3  # HAVE TO SET HIGHER Z VALUE TO AVOID COLLISION WITH THE GROUND
            theta = state[2]

            self.robot.set_state([x, y, z], theta)
            mujoco.mj_forward(self.robot.model, self.robot.data)

            # no contact -> valid
            return self.robot.is_collid() == False  
        

        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
        self.si.setup()

        ## set start and goal
        self.start = ob.State(state_space)
        self.goal = ob.State(state_space)
        for i in range(3):
            self.start[i] = start[i]
            self.goal[i] = goal[i]


    # solve
    def solve(self):
        pdef = ob.ProblemDefinition(self.si)
        pdef.setStartAndGoalStates(self.start, self.goal)
        planner = og.RRTConnect(self.si)
        planner.setRange(0.01)
        planner.setProblemDefinition(pdef)
        planner.setup()
        solved = planner.solve(10.0)
        self.path_states = []
        if solved:
            self.path = pdef.getSolutionPath()
            for i in range(self.path.getStateCount()):
                state = self.path.getState(i)
                state_values = [state[i] for i in range(3)]
                self.path_states.append(state_values)
                print(state_values)
        else:
            print("No solution found.")
        self.index = 0
        return solved, self.path_states

        
