import mujoco
import mujoco.viewer
import numpy as np


class MujocoViewer():
    def __init__(self, mujoco_model, mujoco_data):
        """
        mujoco_model: model in mujoco you want to show 
        mujoco_data: data of the corresponding model
        """
        self.model = mujoco_model
        self.data = mujoco_data

        if self.model is None or self.data is None:
            raise ValueError("[MujocoViewer]: model or data cannot be None")

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.ngeo = 0

        self.max_geoms = len(self.viewer.user_scn.geoms)

    def set_camera(self, distance=None, azimuth=None, elevation=None, lookat=None):
        """
        Set the camera view parameters.
        
        Args:
            distance: Distance from the camera to the lookat point
            azimuth: Horizontal rotation angle in degrees (0-360)
            elevation: Vertical angle in degrees (-90 to 90)
            lookat: 3D point [x, y, z] that the camera looks at
        """
        if not self.viewer.is_running():
            raise RuntimeError("[MujocoViewer]: Viewer window is closed")
        
        cam = self.viewer.cam
        
        if distance is not None:
            cam.distance = distance
        if azimuth is not None:
            cam.azimuth = azimuth
        if elevation is not None:
            cam.elevation = elevation
        if lookat is not None:
            cam.lookat[:] = lookat


    def draw_point(self, pt, size=0.05, rgba=np.array([1, 0, 0, 1])):
        """
        pt: [x, y, z]
        size: size of marker
        rgba: color [r, g, b, alpha]
        """
        if self.viewer.is_running():

            if self.ngeo >= self.max_geoms:
                print(f"[MujocoViewer]: Exceeded max_geoms={self.max_geoms}, resetting to 0.")
                self.reset()


            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[self.ngeo],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[size, 0, 0],
                pos=np.array([pt[0], pt[1], pt[2]]),
                mat=np.eye(3).flatten(),
                rgba=rgba
            )
            self.ngeo += 1
            self.viewer.user_scn.ngeom = self.ngeo
        else:
            raise RuntimeError("[MujocoViewer]: Viewer window is closed")
        

    def draw_line_segment(self, pt_from, pt_to, width=0.001, rgba=np.array([1, 0, 0, 1])):
        """
        pt_from: 1 end point of line segment, [x,y,z]
        pt_to: the other end point, [x,y,z]
        width: width of the line
        rgba: color
        """
        if self.viewer.is_running():

            if self.ngeo >= self.max_geoms:
                print(f"[MujocoViewer]: Exceeded max_geoms={self.max_geoms}, resetting to 0.")
                self.reset()

            mujoco.mjv_connector(
                self.viewer.user_scn.geoms[self.ngeo],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                width=width,
                from_=np.array([pt_from[0], pt_from[1], pt_from[2]]),
                to=np.array([pt_to[0], pt_to[1], pt_to[2]])
            )

            self.viewer.user_scn.geoms[self.ngeo].rgba = rgba
            self.ngeo += 1
            self.viewer.user_scn.ngeom += self.ngeo
        else:
            raise RuntimeError("[MujocoViewer]: Viewer window is closed")



    def draw_traj(self, traj, size=0.05, rgba=np.array([1, 0, 0, 1])):
        """
        traj: Nx3 array
        """
        
        if self.viewer.is_running():
            for pt in traj:
                self.draw_point(pt, size=size, rgba=rgba)
        else:
            raise RuntimeError("[MujocoViewer]: Viewer window is closed")
    


    def render(self):
        """
        show in mujoco
        """
        if self.viewer.is_running():
            self.viewer.sync()
        else:
            raise RuntimeError("Viewer window is closed")


    def close(self):
        """
        close it
        """
        self.viewer.close()


    def reset(self):
        """
        need to reset if the current number of geom > max
        """
        self.viewer.user_scn.ngeom = 0
        self.ngeo = 0

    
    def reset(self, ngeom):
        """
        instead of directly setting to 0, we can start to overwrite from specific position in the list

        """
        self.viewer.user_scn.ngeom = ngeom
        self.ngeo = ngeom


    def is_running(self):
        return self.viewer.is_running()