"""
2D grid map class, used for path planning and obstacle avoidance.
"""

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

class GridMap:
    def __init__(self, model, data, resolution, width, height, robot_radius, margin):
        """
        :param model: MuJoCo MjModel object
        :param data: MuJoCo MjData object
        :param resolution: grid resolution
        :param width: width of the gridmap (in meters)
        :param height: height of the gridmap (in meters)
        :param robot_radius: radius of the robot (in meters)
        :param margin: safety margin around obstacles (in meters)
        """
        self.model = model
        self.data = data
        self.resolution = resolution
        self.width = width
        self.height = height
        self.grid_width = int(width / self.resolution)
        self.grid_height = int(height / self.resolution)

        self.robot_radius = robot_radius
        self.margin = margin
        self.inflation_radius = robot_radius + margin

        self.grid = np.zeros((self.grid_height, self.grid_width))
        
        self.create_grid()

    def create_grid(self):
        for i in range(self.model.ngeom):
            geom_type = self.model.geom_type[i]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                self._add_box(i)
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                self._add_sphere(i)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                self._add_cylinder(i)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                self._add_capsule(i)
            else:
                pass


    def _add_box(self, geom_id):
        
        # box center
        center = self.data.geom_xpos[geom_id]
        lx, ly, lz = self.model.geom_size[geom_id]
        R = self.data.geom_xmat[geom_id].reshape(3, 3)

        # 4 cornors in box coordinate
        local_pts = np.array([
            [-lx / 2 - self.inflation_radius, -ly / 2 - self.inflation_radius],
            [ lx / 2 + self.inflation_radius, -ly / 2 - self.inflation_radius],
            [ lx / 2 + self.inflation_radius,  ly / 2 + self.inflation_radius],
            [-lx / 2 - self.inflation_radius,  ly / 2 + self.inflation_radius]
        ])

        # to world coordinate
        world_pts = np.dot(local_pts, R[:2, :2].T) + center[:2] 

        # mark grid inside obs
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                x = (i + 0.5) * self.resolution  
                y = (j + 0.5) * self.resolution  

                if self._point_in_polygon(np.array([x, y]), world_pts):
                    self.grid[j, i] = 1

    def _add_sphere(self, geom_id):
        
        center = self.data.geom_xpos[geom_id]
        radius = self.model.geom_size[geom_id][0] 

        for i in range(self.grid_width):
            for j in range(self.grid_height):
                x = (i + 0.5) * self.resolution
                y = (j + 0.5) * self.resolution
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if dist <= radius + self.inflation_radius:
                    self.grid[j, i] = 1

    def _add_cylinder(self, geom_id):
        center = self.data.geom_xpos[geom_id]
        radius = self.model.geom_size[geom_id][0]  
        height = self.model.geom_size[geom_id][1] 

        for i in range(self.grid_width):
            for j in range(self.grid_height):
                x = (i + 0.5) * self.resolution
                y = (j + 0.5) * self.resolution
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if dist <= radius + self.inflation_radius:
                    self.grid[j, i] = 1


    def _point_in_polygon(self, point, polygon):
        """
        Ray-casting algorithm to determine if point is in polygon
        """
        n = len(polygon)
        inside = False
        x, y = point
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    

    def coor_to_index(self, coor):
        x, y = coor[0], coor[1]
        col = int((x + self.resolution / 2) / self.resolution)
        row = int((y + self.resolution / 2) / self.resolution)
        return row, col

    def index_to_coor(self, ind):
        row, col = ind[0], ind[1]
        x = (col - self.resolution / 2) * self.resolution
        y = (row - self.resolution / 2) * self.resolution
        return x, y
    

    def is_valid_index(self, index):
        row, col = index
        return 0 <= row < self.grid_height and 0 <= col < self.grid_width
    
    def is_occupied_index(self, index):
        row, col = index
        return self.grid[row, col] == 1


    def show_map(self):
        plt.imshow(self.grid, cmap='gray')
        plt.title("2D Grid Map with Obstacles")
        plt.show()


    def draw_path(self, model, data, path, radius=0.01, color=[1, 0, 0]):
        for i, (x, y) in enumerate(path):
            # 在路径点处添加一个小球
            geom_name = f"path_point_{i}"
            
            # 创建一个小球（sphere）来表示路径点
            # x, y 表示坐标，radius 表示半径，z 坐标可以设置为路径点的高度（默认为 0）
            geom = mujoco.MjsGeom(name=geom_name, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius])
            
            # 设置路径点的位置（我们假设路径在 2D 平面内，z 坐标为 0）
            geom.pos = np.array([x, y, 0.0])
            
            # 将该小球添加到模型的 worldbody 中
            model.worldbody.append(geom)
            
            # 设置颜色
            model.geom_rgba.append(color)
            
        return model, data
