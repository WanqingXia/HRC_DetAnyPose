#!/usr/bin/env python
#!\file
#
# \author  Wanqing Xia wxia612@aucklanduni.ac.nz
# \date    2022-07-28
#
#
# ---------------------------------------------------------------------

import sys
import rospy
import time

from yolo_msgs.msg import yolo

class AzureKinectCamera:
    def __init__(self):
        self.listener = rospy.Subscriber("/yolov5", yolo, self._update_object_stat, queue_size=10)
        self._detect = yolo()

    def _update_object_stat(self, stat):
        self._detect = stat
    
    def get_detect(self):
        return self._detect