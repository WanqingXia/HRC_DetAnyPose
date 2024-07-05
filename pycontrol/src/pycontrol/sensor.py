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

from robotiq_ft_sensor.msg import ft_sensor

class FT300Sensor:
    def __init__(self):
        self.listener = rospy.Subscriber("/robotiq_ft_sensor", ft_sensor, self._update_sensor_stat, queue_size=10)
        self._reading = ft_sensor()

    def _update_sensor_stat(self, stat):
        self._reading = stat
    
    def get_reading(self):
        return self._reading