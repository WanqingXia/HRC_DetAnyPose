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

from robotiq_85_msgs.msg import GripperCmd, GripperStat


class Robotiq85Gripper:

    def __init__(self):

        rospy.Subscriber("/gripper/stat", GripperStat, self._update_gripper_stat, queue_size=10)
        self._gripper_pub = rospy.Publisher('/gripper/cmd', GripperCmd, queue_size=10)      

            
        self._gripper_stat = GripperStat()
        self._gripper_cmd = GripperCmd()
        self._r = rospy.Rate(1)
        self.open()
            
    def _update_gripper_stat(self, stat):
        self._gripper_stat = stat

    def close(self):
        for i in range(10):
            if self._gripper_stat.is_ready:
                for j in range(10):
                    if self._gripper_stat.is_moving:
                        rospy.loginfo('Gripper is moving, retrying i%/10', (j))
                        self._r.sleep()
                        time.sleep(0.5)
                    else:
                        self._gripper_cmd.position = 0.0
                        self._gripper_cmd.speed = 0.02
                        self._gripper_cmd.force = 100.0
                        self._gripper_pub.publish(self._gripper_cmd)
                        return True
                return False
            else:
                rospy.loginfo('Gripper is not ready, retrying %2d/10' % (i))
                self._r.sleep()
                time.sleep(0.5)
        return False

    def open(self):
        for i in range(10):
            if self._gripper_stat.is_ready:
                for j in range(10):
                    if self._gripper_stat.is_moving:
                        rospy.loginfo('Gripper is moving, retrying %2d/10' % (j))
                        self._r.sleep()
                        time.sleep(0.5)
                    else:
                        self._gripper_cmd.position = 0.085
                        self._gripper_cmd.speed = 0.02
                        self._gripper_cmd.force = 100.0
                        self._gripper_pub.publish(self._gripper_cmd)
                        return True
                return False
            else:
                rospy.loginfo('Gripper is not ready, retrying %2d/10' % (i))
                self._r.sleep()
                time.sleep(0.5)
        return False

    def get_stat(self):
        return self._gripper_stat
