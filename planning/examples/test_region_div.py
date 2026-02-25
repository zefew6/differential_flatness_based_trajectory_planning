"""
Divide the whole working region based on the workload for each robot and then assign the sub-region to each robot.
"""


import mujoco
import time
import numpy as np
from matplotlib.patches import Wedge
from matplotlib.path import Path
from scipy.spatial import cKDTree

from m0.sensor.camera.camera import Camera
from m0.control.keyboard import KeyboardController
from m0.control.pid import PIDController
from m0.viewer.mujoco_visualization import MujocoViewer
from m0.robot.robot import Robot
from m0.utils.utils import wrap_pi

from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment




# ----------------------- utils functions -------------------------

def get_center_pt(sample_pts):
    """
    get the mass center of the sample pts on the contour
    sample_pts: Nx2 array, assume the mass are the same on each point
    """
    return np.mean(sample_pts, axis=0)


def partition_by_workload(num_rob, pts_on_contour, par_outline):
    """
    partition the region by workload
    num_rob: number of robots
    pts_on_contour: (N,2) array, points on the target contour
    par_outline: (M,2) array, points representing the particles outside the target contour
    """
    N = pts_on_contour.shape[0]
    workload = np.zeros((N,))  # workload on each contour point
    # for each particle outline, find the closest point on the contour
    tree = cKDTree(pts_on_contour)
    for par in par_outline:
        dists, idxs = tree.query(par, k=50)
        sigma = 0.6 
        weights = np.exp(-0.5 * (dists/sigma)**2)
        weights /= weights.sum() + 1e-12  # normalize
        workload[idxs] += dists * weights  # increase the workload count

    total_workload = workload.sum()
    workload_per_robot = total_workload / num_rob

    ## assign contour points to robots using searchsort
    assignment_idx = [0]   # start point
    cumulative_workload = 0.0
    current_robot = 0
    for i in range(N):
        cumulative_workload += workload[i]
        if cumulative_workload >= workload_per_robot * (current_robot+1) and current_robot < num_rob-1:
            assignment_idx.append(i)
            current_robot += 1

    return assignment_idx, workload



def points_in_contour(particle_pts, contour_pts):
    """
    particle_pts: (M,2) array
    contour_pts:  (K,2) array, closed contour (last point == first point not required)
    return: mask (M,) bool array, True if inside
    """
    path = Path(contour_pts, closed=True)
    return path.contains_points(particle_pts)



def hungarian_algorithm(robot_positions, center, sector_edges_deg, r_assign=3.0):
	"""
	robot_positions: (R,2) array, robot positions in world coordinate
	center: (2,) array, center point of the target shape
	sector_edges_deg: (num_sectors+1,) array, sector edges in degree, e.g. [10, 90, 170, 250, 330, 370]
	r_assign: float, the distance from center to the base point in each sector
	"""

	num_robots = robot_positions.shape[0]
	num_sectors = len(sector_edges_deg) - 1
	assert num_robots == num_sectors, "Number of robots must equal number of sectors"

	sector_points = []
	for i in range(num_sectors):
		mid_angle = np.deg2rad(0.5 * (sector_edges_deg[i] + sector_edges_deg[i+1]))
		px = center[0] + r_assign * np.cos(mid_angle)
		py = center[1] + r_assign * np.sin(mid_angle)
		sector_points.append((px, py))
	sector_points = np.array(sector_points)

	cost_matrix = np.linalg.norm(
		robot_positions[:,None,:] - sector_points[None,:,:], axis=2
	)  # shape (R, num_sectors)

    
	row_ind, col_ind = linear_sum_assignment(cost_matrix)

	return row_ind, col_ind, sector_points, cost_matrix






if __name__ == "__main__":

	# load the model
	model = mujoco.MjModel.from_xml_path("m0/assets/scene_p170_r5.xml")
	data = mujoco.MjData(model)

	# init the viewer
	mjv = MujocoViewer(model, data)

	# init the camera
	cam = Camera(model, data, cam_name="topdown", width=640, height=480)


	# create the target contour
	t = np.linspace(0, 2*np.pi, 300, endpoint=False)
	# ellipse shape
	target_pts = np.vstack((np.cos(t), 0.5 * np.sin(t))).T
	target_pts_xyz = np.hstack((target_pts, 0.03 * np.ones((target_pts.shape[0], 1))))
	

	# get the center point of the target shape
	center_pt = get_center_pt(target_pts)
	cx, cy = center_pt


	# create robots and controller instances
	num_robots = 5
	robots = []
	robot_controllers = []
	for i in range(num_robots):
		robot = Robot(model, data, robot_body_name=f"{i}", 
				actuator_names=[f"forward_{i}", f"turn_{i}"], max_v=1.0, max_w=1.0)
		robots.append(robot)
		robot_controller = PIDController(np.zeros(2)) # temporary goal [0,0]
		robot_controllers.append(robot_controller)

	

	t = 0
	cal_time = 5000
	pic_time = 5

	# -------- loop --------
	while mjv.is_running():

		if t % cal_time == 0 and t == 0: # only cal once at the begining

			# reset the geomtery
			mjv.reset(0)
			print("------- new division cal -------")

			# take picture
			img = cam.capture()

			# get the gray image using hsv threshold
			mask = cam.get_grayimg(img, 
									np.array([140, 50, 50]), 
									np.array([170, 255, 255]))
			
			# get the pixel's world coord outside the target shape
			particle_pixel = np.column_stack(np.where(mask >= 250))
			### exchange the col and row to x and y: (row, col) -> (x=col, y=row)
			particle_pixel = particle_pixel[:, ::-1]
			particle_pts = cam.img_2_world(particle_pixel)


			# divide the region by workload
			assignment_idx, workload = partition_by_workload(num_robots, target_pts, particle_pts)
			assignment_idx = sorted(assignment_idx)
			assignment_pts = target_pts[assignment_idx]


			# hungrarian algo
			angles = np.rad2deg(np.arctan2(assignment_pts[:,1]-cy, assignment_pts[:,0]-cx)) % 360
			order = np.argsort(angles)
			sorted_angles = angles[order]
			sector_edges = np.append(sorted_angles, sorted_angles[0]+360)
			robot_xy = np.zeros((num_robots, 2))
			for i in range(num_robots):
				robot_xy[i] = robots[i].get_pos()[:2].copy()
			row_ind, col_ind, region_base, _ = hungarian_algorithm(robot_xy, center_pt, sector_edges)


			# set the goal for each robot
			for r, s in zip(row_ind, col_ind):
				yaw = robots[r].get_state()[2]
				thr = np.arctan2(cy-region_base[s, 1],
								cx-region_base[s, 0])
				goal = np.array([region_base[s, 0], 
					 			region_base[s, 1], 
								thr])
				robot_controllers[r].set_goal(goal)


			# draw tartget shape
			mjv.draw_traj(target_pts_xyz, size=0.01, rgba=[0,0,1,1])

			# draw assgiment point and the sector region boundry
			pt_from = np.array([center_pt[0], center_pt[1], 0.04])
			pt_to = np.array([5 * (assignment_pts[:,0] - center_pt[0]), 
					          5 * (assignment_pts[:,1] - center_pt[1]), 
							0.04 * np.ones((assignment_pts.shape[0],))]).T
			for i in range(len(assignment_pts)):
				tform = np.array([assignment_pts[i,0], assignment_pts[i,1], 0.04])
				mjv.draw_point(tform, size=0.05, rgba=[0,1,0,1])
				mjv.draw_line_segment(pt_from, pt_to[i], width=0.001 , rgba=[1,1,0,1])

			# draw assigned sector region
			for r, s in zip(row_ind, col_ind):
				pt_from = np.array([robots[r].get_pos()[0], robots[r].get_pos()[1], 0.04])
				pt_to = np.array([region_base[s, 0], region_base[s, 1], 0.04])
				mjv.draw_line_segment(pt_from, pt_to, width=0.001, rgba=[1,0,0,1])

			
			
		else:
			# control each robot to move to its goal
			for i in range(num_robots):
				pos_yaw = robots[i].get_state() # [x, y, yaw]
				vel6 = robots[i].get_v() #[vx, vy, vz, wx, wy, wz]
				lin_v, ang_w = vel6[:3], vel6[3:]
				state = np.array([pos_yaw[0], pos_yaw[1], pos_yaw[2],
								lin_v[0], lin_v[1], ang_w[2]])
				
				v, w = robot_controllers[i].step(state)
				robots[i].set_ctrl(v, w)

			mujoco.mj_step(model, data)

		if t % pic_time == 0:
			mjv.render()

		time.sleep(0.0002)

		t += 1