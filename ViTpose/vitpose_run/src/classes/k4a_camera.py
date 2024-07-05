"""
k4a_camera.py

Author: Wanqing Xia
Email: wxia612@aucklanduni.ac.nz

This script contains the k4a_camera class which is used for reading image data from Azure Kinect camera.
"""

# This will import all the public symbols into the pykinect_azure namespace.
import pykinect_azure as pykinect

class K4A_Camera:
    def __init__(self):
        # Initialize the library, if the library is not found, add the library path as argument
        pykinect.initialize_libraries()

        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

        # Start device
        self.device = pykinect.start_device(config=device_config)

    def get_calibration(self):
        return self.device.calibration

    def get_capture(self):
        capture = self.device.update()

        # Get the color image from the capture
        ret_color, color_image = capture.get_color_image()

        # Get the colored depth
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

        return ret_color, color_image, ret_depth, transformed_depth_image

    def __del__(self):
        self.device.close()

