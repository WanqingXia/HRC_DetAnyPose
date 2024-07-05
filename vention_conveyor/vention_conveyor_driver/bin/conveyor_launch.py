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

from vention_conveyor_driver.MachineMotion import *
from vention_conveyor_msgs.msg import ConveyorCmd, ConveyorStat
from sensor_msgs.msg import JointState

class ConveyorDriver:
    
    def __init__(self):
        self._minpos = rospy.get_param('~min_position', 0)
        self._maxpos = rospy.get_param('~max_position', 710)
        self._ip = rospy.get_param('~ip', "192.168.12.200")
        self._acceleration = rospy.get_param('~init_acceleration', 50)
        self._speed = rospy.get_param('~init_speed', 50)
        self._position = 0.0
        self.is_ready = True
        self._conveyor = MachineMotion(machineIp=self._ip)

        # Configure the actuator
        self._axis = 1
        self._conveyor.configAxis(self._axis, MICRO_STEPS.ustep_8, MECH_GAIN.rack_pinion_mm_turn)

        self._start()

        rospy.Subscriber("/conveyor/cmd", ConveyorCmd, self._update_conveyor_cmd, queue_size=10)
        self._conveyor_pub = rospy.Publisher('/conveyor/stat', ConveyorStat, queue_size=10)
        self._conveyor_joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)

        self._run_driver()

    def _start(self):
        self._conveyor.setAcceleration(self._acceleration)
        rospy.loginfo("Conveyor acceleration set to " + str(self._acceleration) + "mm/s^2.")
        self._conveyor.setSpeed(self._speed)
        rospy.loginfo("Conveyor speed set to " + str(self._speed) + "mm/s.")
        # When starting a program, one must remove the software stop before moving
        rospy.loginfo("Removing software stop for conveyor belt")
        self._conveyor.releaseEstop()
        rospy.loginfo("Resetting conveyor belt")
        self._conveyor.resetSystem()

        self._conveyor.moveToHome(self._axis)
        rospy.loginfo("Sending conveyor belt to home position")
        self._conveyor.waitForMotionCompletion()
        pos_home = self._conveyor.getActualPositions(self._axis)
        rospy.loginfo("the position is " + str(pos_home))
        if pos_home > 1 or pos_home < 0:
            self.is_ready &= False

        # position_desired = 100
        # self._conveyor.moveToPosition(self._axis, position_desired)
        # rospy.loginfo("Sending conveyor belt to 100mm from home position")
        # self._conveyor.waitForMotionCompletion()
        # pos_actual = self._conveyor.getActualPositions(self._axis)
        # if pos_actual > 101 or pos_actual < 99:
        #     rospy.logerr("Initialisation failed, desired position is 100.0, but actual position is " + str(pos_actual))
        #     self.is_ready &= False
        

    def _update_conveyor_cmd(self, cmd):
        if self.is_ready == True:
            if self._speed != cmd.speed:
                self._speed = cmd.speed
                self._conveyor.setSpeed(self._speed)
                rospy.loginfo("Conveyor speed set to " + str(self._speed) + "mm/s.")
            if self._acceleration != cmd.acceleration:
                self._acceleration = cmd.acceleration
                self._conveyor.setAcceleration(self._acceleration)
                rospy.loginfo("Conveyor acceleration set to " + str(self._acceleration) + "mm/s^2.")

            if cmd.desired_position < self._minpos:
                rospy.loginfo("Minimium position is 0.0mm but got " + str(cmd.desired_position) + "mm.")
                self._conveyor.moveToPosition(self._axis, self._minpos)
                self._conveyor.waitForMotionCompletion()
                self._position = self._conveyor.getActualPositions(self._axis)
                rospy.loginfo("Conveyor moved to " + str(self._position) + "mm from home")

            elif cmd.desired_position > self._maxpos:
                rospy.loginfo("Maximium position is 540.0mm but got " + str(cmd.desired_position) + "mm.")
                self._conveyor.moveToPosition(self._axis, self._maxpos)
                self._conveyor.waitForMotionCompletion()
                self._position = self._conveyor.getActualPositions(self._axis)
                rospy.loginfo("Conveyor moved to " + str(self._position) + "mm from home")

            else:
                self._conveyor.moveToPosition(self._axis, cmd.desired_position)
                self._conveyor.waitForMotionCompletion()
                self._position = self._conveyor.getActualPositions(self._axis)
                rospy.loginfo("Conveyor moved to " + str(self._position) + "mm from home")

    def _update_stat(self):
        stat = ConveyorStat()
        self._position = self._conveyor.getActualPositions(self._axis)
        stat.current_position = self._position
        stat.is_ready = self.is_ready
        return stat

    def _run_driver(self):
        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            stat = ConveyorStat()
            stat = self._update_stat()
            self._conveyor_pub.publish(stat)

            state = JointState()
            state.header.stamp = rospy.Time.now()
            state.name = ['workbench_joint']
            state.position = [stat.current_position/1000]
            state.velocity = [0.0]
            state.effort = [0.0]
            self._conveyor_joint_pub.publish(state)

            r.sleep()


if __name__ == "__main__":
    rospy.init_node("converyor_driver")
    ConveyorDriver()
