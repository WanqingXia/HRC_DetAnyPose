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

from vention_conveyor_msgs.msg import ConveyorCmd, ConveyorStat

class ConveyorBelt:
    
    def __init__(self):
        self._stat = ConveyorStat()
        rospy.Subscriber("/conveyor/stat", ConveyorStat, self._update_conveyor_stat, queue_size=10)
        print(self._stat)

        self._cmd = ConveyorCmd()
        self._cmd.acceleration = 50.0
        self._cmd.speed = 50.0   
        self._cmd.desired_position = self._stat.current_position
        self._r = rospy.Rate(1)
        self._conveyor_pub = rospy.Publisher('/conveyor/cmd', ConveyorCmd, queue_size=10)

    def _update_conveyor_stat(self, stat):
        self._stat = stat

    def get_coveyor_stat(self):
        return self._stat
    
    def set_speed(self, speed):
        self._cmd.speed = speed
        self._conveyor_pub.publish(self._cmd)
        rospy.loginfo("Conveyor speed set to " + str(self._cmd.speed) + "mm/s.")
        self._r.sleep()
    
    def set_acceleration(self, acceleration):
        self._cmd.acceleration = acceleration
        self._conveyor_pub.publish(self._cmd)
        rospy.loginfo("Conveyor acceleration set to " + str(self._cmd.acceleration) + "mm/s^2.")
        self._r.sleep()

    def set_position(self, position):
        self._cmd.desired_position = position
        self._conveyor_pub.publish(self._cmd)
        self._r.sleep()
    
    def go_home(self):
        self.set_position(0.0)


