#!/usr/bin/env python
#!\file
#
# \author  Wanqing Xia wxia612@aucklanduni.ac.nz
# \date    2023-08-22
#
#
# ---------------------------------------------------------------------

import sys
import time
import rospy
import numpy as np
import signal

from pycontrol.gripper import Robotiq85Gripper
from pycontrol.sensor import FT300Sensor
from pycontrol.robot import UR5eRobot
from pycontrol.conveyor import ConveyorBelt
from pycontrol.camera import AzureKinectCamera

def sig_handler(signal, frame):
    print("Existing Program...")
    sys.exit(0)

def open_gripper(gripper):
    success = gripper.open()
    if success:
        rospy.loginfo('Successfully opened')
        time.sleep(2)
    else:
        rospy.loginfo('Open gripper failed')
        raise Exception("Cannot open gripper")
    time.sleep(1)

def close_gripper(gripper):
    success = gripper.close()
    if success:
        rospy.loginfo('Successfully closed')
        time.sleep(1)
    else:
        rospy.loginfo('Close gripper failed')
        raise Exception("Cannot close gripper")
    time.sleep(1)

def robot_pick(robot, camera, height, type):
    count = 0
    last_tx, last_ty, last_rot = 0, 0, 0

    
        # only move when count > 6 (object stable for 3 seconds)
    while count < 7:
        detection = camera.get_detect()
        if count == 0:
            if type == "unpacked":
                last_tx = detection.unpacked_tx / 1000
                last_ty = detection.unpacked_ty / 1000
                last_rot = detection.unpacked_rot
            elif type == "packed":
                last_tx = detection.packed_tx / 1000
                last_ty = detection.packed_ty / 1000
                last_rot = detection.packed_rot
            else:
                raise ValueError("The object type is not supported")

        if type == "unpacked":
            dtx = detection.unpacked_tx / 1000
            dty = detection.unpacked_ty / 1000
            drot = detection.unpacked_rot
        elif type == "packed":
            dtx = detection.packed_tx / 1000
            dty = detection.packed_ty / 1000
            drot = detection.packed_rot
        else:
            raise ValueError("The object type is not supported")

        if drot != -100:
            if np.abs(dtx -last_tx) < 0.01 and np.abs(dty-last_ty) < 0.01 and np.abs(drot - last_rot) < 0.05:
                last_tx = dtx
                last_ty = dty
                last_rot = drot
                current_pose = robot.get_actual_pose()
                diag = np.sqrt((current_pose[0] + dtx)**2 + (current_pose[1] + dty)**2)
                # object is too far or may collide with table
                if diag > 0.92:
                    count = 0
                    rospy.loginfo("Object is beyond robot's reach")
                else:
                    # all check pass, plus count
                    count += 1
                    if count == 3:
                        rospy.loginfo("Object stable, confirming pose...")
            else:
                count = 0
                rospy.loginfo("Object is not stable")
        else:
            count = 0
            rospy.loginfo("Object is not stable")

        time.sleep(0.5)

    picking_list = [] # go to picking position
    cp = robot.get_actual_pose()
    cj = robot.get_joint_pose()
    cj_conv = robot.joint_to_cart([cj[0], cj[1], cj[2], cj[3], cj[4], cj[5] + last_rot])
    picking_list.append([cp[0]+ last_tx, cp[1] + last_ty, height, cj_conv[3], cj_conv[4], cj_conv[5]])
    robot.execute_cartesian_trajectory(picking_list)


def wait_movement(robot, conveyor, conveyor_pos):
    while True:
        if np.sum(np.abs(robot.get_pos_error())) < 0.01:
            if np.abs(conveyor.get_coveyor_stat().current_position - conveyor_pos) < 2:
                break
            else:
                time.sleep(0.1)
        else:
            time.sleep(0.1)
    time.sleep(0.5)

if __name__ == "__main__":
    rospy.init_node("combined_control")
    robot = UR5eRobot()
    gripper = Robotiq85Gripper()
    sensor = FT300Sensor()
    conveyor = ConveyorBelt()
    camera = AzureKinectCamera()

    conveyor.set_speed(100)
    conveyor.set_acceleration(100)

    signal.signal(signal.SIGINT, sig_handler)

    # send robot to home position
    robot.go_home()

    # opening gripper
    open_gripper(gripper)

    while True:
        # turn over to conveyor side
        pose_list = []
        pose_list.append([0.297, -0.132, 0.272, 2.226, -2.217, 0.0])
        pose_list.append([-0.132, -0.297, 0.272, 0.0, -3.141, 0.0])
        robot.execute_cartesian_trajectory(pose_list)

        # send converyor to home position
        conveyor.go_home()

        over_list =[]
        over_list.append([-0.191, -0.668, 0.250, 0.0, -3.141, 0.0])
        robot.execute_cartesian_trajectory(over_list)

        # block further movement until conveyor finished moving
        wait_movement(robot, conveyor, 0)

        robot_pick(robot, camera, 0.08, "unpacked")
        
        # close gripper
        close_gripper(gripper)

        conveyor.set_position(270)

        handover_list = []
        handover_list.append([-0.132, -0.78, 0.35, 0.0, -3.141, 0.0])
        handover_list.append([-0.132, -0.297, 0.272, 0.0, -3.141, 0.0])
        handover_list.append([0.327, -0.15, 0.257, 2.479, -2.541, 1.440])
        robot.execute_cartesian_trajectory(handover_list)

        wait_movement(robot, conveyor, 270)

        sensor_reading = sensor.get_reading()
        while True:
            new_reading = sensor.get_reading()
            if abs(new_reading.Fx - sensor_reading.Fx) > 9 or abs(new_reading.Fy - sensor_reading.Fy) > 5 or abs(new_reading.Fz - sensor_reading.Fz) > 8:
                print(new_reading.Fx - sensor_reading.Fx, new_reading.Fy - sensor_reading.Fy, new_reading.Fz - sensor_reading.Fz)
                # open gripper
                open_gripper(gripper)
                break
            else:
                pass

        
        observe_list = []
        observe_list.append([0.586, -0.132, 0.663, 2.227, -2.217, 0.0])
        robot.execute_cartesian_trajectory(observe_list)


        robot_pick(robot, camera, 0.135, "packed")

        # close gripper
        close_gripper(gripper)

        place_list= []
        place_list.append([0.297, -0.132, 0.348, 2.226, -2.217, 0.0])
        place_list.append([0.077, 0.656, 0.200, 2.176, -2.267, 0.0])
        robot.execute_cartesian_trajectory(place_list)

        # open gripper
        open_gripper(gripper)

        robot.go_home()
        conveyor.go_home()
        wait_movement(robot, conveyor, 0)
        time.sleep(2)
