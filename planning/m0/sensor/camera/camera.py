"""
Camera class, to capture images from Mujoco simulation
"""

import mujoco
import numpy as np
import cv2

from m0.sensor.camera.cali import PixelMapper


class Camera:
    def __init__(self, model, data, cam_name, width=640, height=480):
        """
        Camera class, to capture images from Mujoco simulation
        model: mujoco model
        data: mujoco data
        cam_name: name of the camera in the mujoco model
        width: image width
        height: image height
        """
        self.model = model
        self.data = data
        self.cam_name = cam_name
        self.width = width
        self.height = height

        self.renderer = mujoco.Renderer(model, width=self.width, height=self.height)

        # pixel to world transformer, note that you should calibrate the camera first
        self.pixel_mapper = PixelMapper(img_width=self.width, img_height=self.height)
        


    def get_grayimg(self, path, 
                    hsv_low=np.array([140, 50, 50]), 
                    hsv_high=np.array([170, 255, 255])):
        """
        read image from path and get the gray image using hsv threshold
        """
        img = cv2.imread(path)
        return self.get_grayimg(img, hsv_low, hsv_high)


    def get_grayimg(self, img, 
                    hsv_low=np.array([140, 50, 50]), 
                    hsv_high=np.array([170, 255, 255])):
        """
        transfer from RGB image to HSV then
        get the gray image using hsv threshold
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        return mask


    def norm_pixel(self, p):
        """
        pixel to coordinate in [0,1]
        """
        normalized_points = np.zeros_like(p, dtype=float)
        normalized_points[:,0] = p[:,1] / self.width
        normalized_points[:,1] = p[:,0] / self.height
        return normalized_points


    def denorm_pixel(self, coord):
        """
        coordinate in [0,1] to pixel
        """
        pixel = coord * [self.width, self.height]
        return pixel
    

    def img_2_world(self, pixel):
        """
        pixel to world coordinate
        """
        return self.pixel_mapper.pixel_to_realworld(pixel)

    def world_2_img(self, world_coord):
        """
        world coordinate to pixel
        """
        pixel = self.pixel_mapper.realworld_to_pixel(world_coord)
        return pixel

    def capture(self):
        """
        Capture an image from the camera.
        """
        self.renderer.update_scene(self.data, camera="topdown")
        img = self.renderer.render()
        return img