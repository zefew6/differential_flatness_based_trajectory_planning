from xml.parsers.expat import model
import mujoco
import numpy as np

class Robot:
    def __init__(self, model, data, robot_body_name, actuator_names, max_v=1.0, max_w=1.0):
        """
        model: MuJoCo model, may contain multiple robots
        data: MuJoCo data, may contain multiple robots info
        robot_body_name: name of a robot body in XML (for position extraction)
        actuator_names: names of the actuators for this robot: ["actuator1", "actuator2"]
        max_v: max linear velocity v \in [-max_v, max_v]
        max_w: max angular velocity w \in [-max_w, max_w]
        """
        self.model = model
        self.data = data

        self.max_v = max_v
        self.max_w = max_w

        # get robot body id
        self.robot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_body_name)
        if self.robot_id == -1:
            raise ValueError(f"[Robot] Cannot find body '{robot_body_name}' in model.")
        
        # joint id of the robot body
        jnt_id = self.model.body_jntadr[self.robot_id]
        self.adr = self.model.jnt_qposadr[jnt_id]

        # read actuators
        self.ctrl_idx = []
        for name in actuator_names:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) == -1:
                raise ValueError(f"[Robot] Cannot find actuator '{name}' in model.")
            self.ctrl_idx.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))   
    

    def get_pos(self):
        """Return [x, y, z] world position"""
        return self.data.xpos[self.robot_id].copy()

    def get_yaw(self):
        """Return yaw angle (rotation around z)"""
        quat = self.data.xquat[self.robot_id].copy()
        w, x, y, z = quat
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return yaw

    def get_state(self):
        """Return [x, y, yaw] state"""
        pos = self.get_pos()
        yaw = self.get_yaw()
        return np.array([pos[0], pos[1], yaw])
    

    def get_v(self):
        """
        return body's [vx, vy, vz, wx, wy, wz]
        """
        vel6 = self.data.cvel[self.robot_id].copy()
        return vel6
    
    def get_ctrl(self):
        return np.array([self.data.ctrl[self.ctrl_idx[0]].copy(), 
                         self.data.ctrl[self.ctrl_idx[1]].copy()])
    

    def set_ctrl(self, v, w):
        """
        send ctrl cmd to robot
        v: linear velocity
        w: angular velocity
        """
        self.data.ctrl[self.ctrl_idx[0]] = v
        self.data.ctrl[self.ctrl_idx[1]] = w


    def set_pos(self, xyz):
        """
        set robot position (x,y,z)
        ================================
        WARNING: mj_forward must be called after this to update the sim state
        ================================
        """
        self.data.qpos[self.adr:self.adr+3] = np.array([xyz[0], xyz[1], xyz[2]])
    

    def set_yaw(self, yaw):
        """
        set robot yaw angle
        ================================
        WARNING: mj_forward must be called after this to update the sim state
        ================================
        """
        qw = np.cos(yaw / 2.0)
        qz = np.sin(yaw / 2.0)
        self.data.qpos[self.adr+3:self.adr+7] = np.array([qw, 0.0, 0.0, qz])

    
    def set_state(self, xyz, yaw):
        """
        set robot state [x,y,yaw]
        z: height of robot
        ================================
        WARNING: mj_forward must be called after this to update the sim state
        ================================
        """
        self.set_pos(xyz)
        self.set_yaw(yaw)


    def is_collid(self):
        """
        check if the robot is in collision (if has contact)
        """
        return self.data.ncon > 0
    


    def continous_dyn(self):
        """
        continous dynamics of the robot
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = w
        """
        state = self.get_state()
        v, w = self.get_ctrl()

        x_dot = v * np.cos(state[2])
        y_dot = v * np.sin(state[2])
        theta_dot = w

        return np.array([x_dot, y_dot, theta_dot])
    

    def discrete_dyn(self, state, control, dt):
        """
        discrete dynamics of the robot
        state: [x, y, theta]
        control: [v, w]
        dt: time step
        """
        x, y, theta = state
        v, w = control

        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + w * dt

        return np.array([x_new, y_new, theta_new])

    
