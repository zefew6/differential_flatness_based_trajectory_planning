import mujoco
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from shapely.geometry import LineString

from m0.sensor.camera.camera import Camera
from m0.control.keyboard import KeyboardController
from m0.viewer.mujoco_visualization import MujocoViewer
from m0.robot.robot import Robot

from concave_hull import (
    concave_hull,
    concave_hull_indexes,
    convex_hull_indexes,
)




# ----------------------- utils functions -------------------------


def get_concave_contour(norm_coord, length_threshold):
    """
    get the concave contour of the mass of points
    """
    idxes = concave_hull_indexes(
        norm_coord[:, :2],
        length_threshold=length_threshold,
        # convex_hull_indexes=convex_hull.vertices.astype(np.int32),
    )
    return norm_coord[idxes]


def sample_concave_contour_points(vertex, N):
    """
    get a set of points sampled on the concave contour
    """
    line = LineString(vertex)
    length = line.length
    # sample N
    distances = np.linspace(0, length, N, endpoint=False)
    sampled_points = np.array([list(line.interpolate(d).coords)[0] for d in distances])
    return sampled_points



def recons_contour_fft(mask, cam, H, N):
    """
    reconstruct contour using FFT, parametrized by s_tilde~[0,1], for more detail see:
    "Fourier-Based Multi-Agent Formation Control to Track Evolving Closed Boundaries"

    mask: gray image
    cam: camera instance
    H: fourier hamonic
    N: num of sample points on contour
    """

    I = np.eye(2)
    height, width = mask.shape[0], mask.shape[1]
    # check if the height and width of mask match the camera
    assert height == cam.height and width == cam.width, "the height and width of mask should match the camera"

    # get the points on concave contour
    particle_pixel = np.column_stack(np.where(mask >= 250))
    norm_coord = cam.norm_pixel(particle_pixel)
    vertex = get_concave_contour(norm_coord, 0.001)
    sample_points = sample_concave_contour_points(vertex, N)
    C = sample_points.flatten()


    s_tilde = np.linspace(0, 1, N)  # parameter
    # compute fourier coefficients

    # -----------  get gh ------------
    Big_G = np.zeros((4*H + 2, 2 * N))
    for i in range(N):
        G = np.zeros((2, 4*H + 2))
        for j in range(H):
            gh = np.array([[np.cos(2 * np.pi * (j + 1) * s_tilde[i]), np.sin(2 * np.pi * (j + 1) * s_tilde[i]), 0, 0],
                        [0, 0, np.cos(2 * np.pi * (j + 1) * s_tilde[i]), np.sin(2 * np.pi * (j + 1) * s_tilde[i])]])
            G[:, 4*j:4*j+4] = gh
        G[:, -2:] = I

        Big_G[:, 2*i:2*i+2] = G.T

    theta, _, _, _ = np.linalg.lstsq(Big_G.T, C, rcond=None)


    # ---------- reconstruct the parametrized contour using fft coefficients ------------
    recon_coord = np.zeros((N, 2))
    e, f = theta[-2], theta[-1]
    for t in range(N):
        c_x_approx, c_y_approx = 0, 0
        for h in range(1, H + 1):
            ah = theta[4*(h - 1)] # x cos
            bh = theta[4*(h - 1) + 1] # x sin
            ch = theta[4*(h - 1) + 2] # y cos
            dh = theta[4*(h - 1) + 3] # y sin

            c_x_approx += ah * np.cos(2 * np.pi * h * s_tilde[t]) + bh * np.sin(2 * np.pi * h * s_tilde[t])
            c_y_approx += ch * np.cos(2 * np.pi * h * s_tilde[t]) + dh * np.sin(2 * np.pi * h * s_tilde[t])

        c_x_approx += e
        c_y_approx += f
        recon_coord[t, 0] = copy.deepcopy(c_x_approx)
        recon_coord[t, 1] = copy.deepcopy(c_y_approx)

    # denorm to pixel
    recon_pixel = cam.denorm_pixel(recon_coord)
    sample_pixel = cam.denorm_pixel(sample_points)


    return sample_pixel, recon_pixel








# ----------------------- main -------------------------
if __name__ == "__main__":

    # load the model
    model = mujoco.MjModel.from_xml_path("m0/assets/scene_p170_r5.xml")
    data = mujoco.MjData(model)

    # init viewer
    mjv = MujocoViewer(model, data)

    # create camera instance
    cam = Camera(model, data, cam_name="topdown", width=640, height=480)

    # create robot instance
    robot1 = Robot(model, data, robot_body_name="0", actuator_names=["forward_0", "turn_0"], max_v=1.0, max_w=1.0)

    # create keyboard controller
    controller = KeyboardController(robot1.max_v, robot1.max_w)


    # render freq
    render_step = 1000
    t = 0


    while mjv.is_running():

        if t % render_step == 0:

            # reset the geomtery
            mjv.reset(0)

            # get image from camera
            img = cam.capture()

            # get the gray image using hsv threshold
            grayimg = cam.get_grayimg(img, 
                                      np.array([140, 50, 50]), 
                                      np.array([170, 255, 255]))

            # reconstruct the contour using fft
            sample_pixel, recon_pixel = recons_contour_fft(grayimg, cam, H=20, N=150)
            
            # to the world coordinate
            world_coor = cam.img_2_world(recon_pixel)

            # draw in mujoco
            world_coor_xyz = np.hstack((world_coor, 0.03 * np.ones((world_coor.shape[0], 1))))
            mjv.draw_traj(world_coor_xyz)


        # control robot
        v, w = controller.step()
        robot1.set_ctrl(v, w)
        mujoco.mj_step(model, data)

        time.sleep(0.0002)

        mjv.render()
        t += 1





