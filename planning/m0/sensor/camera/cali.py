"""
Code is from: https://github.com/AbhiSharma1999/PixelToRealworld/tree/master

Calibration the camera, the method is to solve the homography matrix using four known points
This method is only valid for points all in the same plane in the real world

"""


import itertools
import numpy as np
import cv2
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt


####  fill up your known four points here, in the order of top-left, 
# top-right, bottom-right, bottom-left
known_realworld_array = np.array([[0.6, 0.6],
                                    [0.6, 1.4],
                                    [1.4, 0.6],
                                    [1.4, 1.4]])


known_pixel_array = np.array([[377.4411255, 181.5588745],
                    [377.4411255, 104.30404051],
                    [454.69595949, 181.5588745],
                    [454.69595949, 104.30404051]])




def compute_camera_matrix(renderer, data, camera_name):
    """Returns the 3x4 camera matrix."""
    # If the camera is a 'free' camera, we get its position and orientation
    # from the scene data structure. It is a stereo camera, so we average over
    # the left and right channels. Note: we call `self.update()` in order to
    # ensure that the contents of `scene.camera` are correct.
    renderer.update_scene(data, camera_name)
    pos = np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
    z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
    y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
    rot = np.vstack((np.cross(y, z), y, z))
    fov = model.vis.global_.fovy

    # Translation matrix (4x4).
    translation = np.eye(4)
    translation[0:3, 3] = -pos

    # Rotation matrix (4x4).
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot

    # Focal transformation matrix (3x4).
    focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * renderer.height / 2.0
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (renderer.width - 1) / 2.0
    image[1, 2] = (renderer.height - 1) / 2.0
    return image @ focal @ rotation @ translation







class PixelMapper(object):
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilteral in both planes
    Parameters
    ----------
    pixel_array : (4,2) shape numpy array
        The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    lonlat_array : (4,2) shape numpy array
        The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    """
    def __init__(self, 
                 pixel_array=known_pixel_array, 
                 realworld_array=known_realworld_array, 
                 img_width=640, 
                 img_height=480):
        assert pixel_array.shape==(4,2), "Need (4,2) input array"
        assert realworld_array.shape==(4,2), "Need (4,2) input array"
        self.M = cv2.getPerspectiveTransform(np.float32(pixel_array),np.float32(realworld_array))
        self.invM = cv2.getPerspectiveTransform(np.float32(realworld_array),np.float32(pixel_array))
        self.img_width = img_width
        self.img_height = img_height

    def pixel_to_realworld(self, pixel):
        """
        Convert a set of pixel coordinates to lon-lat coordinates
        Parameters
        ----------
        pixel : (N,2) numpy array or (x,y) tuple
            The (x,y) pixel coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (lon, lat) coordinates
        """
        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1,2)
        assert pixel.shape[1]==2, "Need (N,2) input array" 
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0],1))], axis=1)
        lonlat = np.dot(self.M,pixel.T)
        
        return (lonlat[:2,:]/lonlat[2,:]).T
    
    def realworld_to_pixel(self, realworld):
        """
        Convert a set of lon-lat coordinates to pixel coordinates
        Parameters
        ----------
        lonlat : (N,2) numpy array or (x,y) tuple
            The (lon,lat) coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (x, y) pixel coordinates
        """
        if type(realworld) != np.ndarray:
            realworld = np.array(realworld).reshape(1,2)
        assert realworld.shape[1]==2, "Need (N,2) input array" 
        realworld = np.concatenate([realworld, np.ones((realworld.shape[0],1))], axis=1)
        pixel = np.dot(self.invM,realworld.T)
        
        return (pixel[:2,:]/pixel[2,:]).T
    








"""
Every time when you move the top-down camera, or when you change the 
picture resolution, width and length and so on, 
Remember to run this to re-calibrate so that the M matrix is correct, and then fill up the M matrix here
"""

if __name__ == "__main__":
    
    #####  create a cali scene in mujoco
    model = mujoco.MjModel.from_xml_path("m0/assets/simple_cali_scene.xml")
    data = mujoco.MjData(model)


    H = 480
    W = 640  # image height and width
    ### The width and height should be the same as your application
    renderer = mujoco.Renderer(model, width=W, height=H)
    mujoco.mj_step(model, data)
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera="topdown")


    ############ get 4 known points in pixel from real world #############

    # Get the world coordinates of the box2 corners
    box_pos = data.geom_xpos[model.geom('box2').id]
    box_mat = data.geom_xmat[model.geom('box2').id].reshape(3, 3)
    box_size = model.geom_size[model.geom('box2').id]
    offsets = np.array([-1, 1]) * box_size[:, None]
    xyz_local = np.stack(list(itertools.product(*offsets))).T
    xyz_global = box_pos[:, None] + box_mat @ xyz_local

    # Camera matrices multiply homogenous [x, y, z, 1] vectors.
    corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
    corners_homogeneous[:3, :] = xyz_global

    # Get the camera matrix.
    m = compute_camera_matrix(renderer, data, camera_name="topdown")

    # Project world coordinates into pixel space. See:
    # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
    xs, ys, s = m @ corners_homogeneous
    # x and y are in the pixel coordinate system.
    x = xs / s
    y = ys / s

    pixel_coor_known_points = np.vstack((x, y)).T
    print(f"known points in the real: {xyz_global}")
    print(f"pixel for known points: {pixel_coor_known_points}")

    # Render the camera view and overlay the projected corner coordinates.
    pixels = renderer.render()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(pixels)
    ax.plot(x, y, '+', c='w')
    ax.set_axis_off()




    ##############  write these into four_knwo_points.py #############
    #### known four points from box2
    realworld_array = np.array([[0.6, 0.6],
                                    [0.6, 1.4],
                                    [1.4, 0.6],
                                    [1.4, 1.4]])
    ######  get the correspoding pixel array from previous calculation above
    pixel_array = np.array([[377.4411255, 181.5588745],
                            [377.4411255, 104.30404051],
                            [454.69595949, 181.5588745],
                            [454.69595949, 104.30404051]])
    
    pw = PixelMapper(pixel_array, realworld_array)

    print("========= The homography matrix M is ============")
    print(pw.M)
    print("========= The inverse matrix M is ===============")
    print(pw.invM)


    


    ##############  Test world to pixel, pick a corner on box1 in the world on the ground (z = 0) #############
    test_world_coord = np.array([[0.4, -0.4]])
    test_pixel_coord = pw.realworld_to_pixel(test_world_coord)
    print(f"the real world coord: {test_world_coord} -> the pixel coord: {test_pixel_coord}")
    ########### if this is correct, then pixel to world should also be correct ##########

    pixels = renderer.render()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(pixels)
    ax.plot(test_pixel_coord[0, 0], test_pixel_coord[0, 1], '+', c='w')
    ax.set_axis_on()
    plt.show()