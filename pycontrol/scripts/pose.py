#!/usr/bin/env python
#!\file
#
# \author  Wanqing Xia wxia612@aucklanduni.ac.nz
# \date    2024-07-02
#
#
# ---------------------------------------------------------------------

import sys
import time
import rospy
import numpy as np
import signal
from scipy.spatial.transform import Rotation as R

from pycontrol.robot import UR5eRobot
from pycontrol.gripper import Robotiq85Gripper
from pycontrol.sensor import FT300Sensor
from geometry_msgs.msg import Pose
from std_msgs.msg import String

#  [ 0.05639628  0.96796691 -0.24365655  0.02215889] translation: [-0.14654684  0.10038842  0.7096086 ]

rotation_from_camera, translation_from_camera = np.zeros(4), np.zeros(3)
object_name = 'none'
pre_rotation_from_camera, pre_translation_from_camera = np.zeros(4), np.zeros(3)
pre_object_name = 'none'
pose_changed, name_changed = False, False

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

def update_object_pose(pose):
    global rotation_from_camera, translation_from_camera, pre_rotation_from_camera, pre_translation_from_camera, pose_changed
    rotation_from_camera = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    translation_from_camera = np.array([pose.position.x, pose.position.y, pose.position.z])
    if not np.array_equal(rotation_from_camera, pre_rotation_from_camera) or not np.array_equal(translation_from_camera, pre_translation_from_camera):
        pre_rotation_from_camera = rotation_from_camera
        pre_translation_from_camera = translation_from_camera
        pose_changed = True

def update_object_name(name):
    global object_name, pre_object_name, name_changed
    object_name = name.data
    if object_name != pre_object_name:
        pre_object_name = object_name
        name_changed = True

def calc_ee_pose(clearance, robot):
    global object_name, rotation_from_camera, translation_from_camera

    object_rotation = R.from_quat(rotation_from_camera).as_matrix()
    # Relative distances from robot end-effector to camera
    ee2camera_dist = np.array([0.035, -0.065, 0.165])  # x, y, z
    # Relative distances from gripper to robot end-effector
    gripper2ee_dist = np.array([0.0, 0.0, -0.245])  # x, y, z

    # ee_link obsevation pose
    ee_current = robot.get_ee_pose()
    ee_translation = ee_current[:3]
    rotation_quat = ee_current[3:]
    ee_rotation = R.from_quat(rotation_quat).as_matrix()

    # Step 1: Calculate the camera world pose based on ee_link pose and relative dist
    camera_translation = ee_translation + ee_rotation @ ee2camera_dist

    # Step 2: Calculate object pose in world frame by its pose in camera, and camera world pose
    object_world_translation = camera_translation + ee_rotation @ translation_from_camera
    object_world_rotation = ee_rotation @ object_rotation

    print(f"object world position: {object_world_translation}")
    print(f"object world rotation: {R.from_matrix(object_world_rotation).as_rotvec()}")

    # Step 3: Make the gripper world pose same as object world pose, then calculate the target pose for ee_link
    # grasp_translation_rel = np.array([0.0, 0.0, -clearance])
    rotation_180 = R.from_euler('x', 180, degrees=True).as_matrix()
    rotation_90 = R.from_euler('z', 90, degrees=True).as_matrix()
    gripper_rotation = object_world_rotation @ rotation_180
    gripper_rotation = gripper_rotation @ rotation_90
    # gripper_translation = object_world_translation + gripper_rotation @ grasp_translation_rel
    gripper_translation = object_world_translation

    # Calculate the ee_link target pose from gripper pose and relative distances
    ee_translation_result = gripper_translation + gripper_rotation @ gripper2ee_dist
    # Convert gripper rotation matrix to rotation vector
    ee_rotation_result = R.from_matrix(gripper_rotation).as_quat()
    return np.hstack((ee_translation_result, ee_rotation_result))


if __name__ == "__main__":
    rospy.init_node("combined_control")
    rospy.Subscriber("/object/pose", Pose, update_object_pose, queue_size=10)
    rospy.Subscriber("/object/name", String, update_object_name, queue_size=10)
    robot = UR5eRobot()
    gripper = Robotiq85Gripper()
    sensor = FT300Sensor()

    signal.signal(signal.SIGINT, sig_handler)
    hand_over_pose = np.array([30, -120, -110, 20, 66, 15])
    rospy.loginfo("Robot waiting for object pose...")
    # opening gripper
    open_gripper(gripper)

    while(True):
        if pose_changed and object_name != "none" and np.sum(translation_from_camera) != 0.0:
            rospy.loginfo(f"{object_name} pose confirmed, translation: {translation_from_camera}, rotation: {rotation_from_camera}")
            clearance = 0.0
            if object_name == "spam rectangular can":
                clearance = 0.1
            elif object_name == "drill":
                clearance = 0.2
            elif object_name == "white bleach bottle":
                clearance = 0.16
            ee_pose = calc_ee_pose(clearance, robot)
            ee_pose[2] = ee_pose[2] + clearance
            robot.execute_cartesian_trajectory(ee_pose)
            ee_pose_down = ee_pose.copy()
            ee_pose_down[2] = ee_pose_down[2] - 0.07
            robot.execute_cartesian_trajectory(ee_pose_down)
            time.sleep(0.1)
            close_gripper(gripper)
            time.sleep(0.1)
            robot.execute_cartesian_trajectory(ee_pose)

            robot.execute_joint_trajectory(hand_over_pose)
            time.sleep(0.1)
            sensor_reading = sensor.get_reading()
            while True:
                new_reading = sensor.get_reading()
                if abs(new_reading.Fx - sensor_reading.Fx) > 9 or abs(new_reading.Fy - sensor_reading.Fy) > 5 or abs(
                        new_reading.Fz - sensor_reading.Fz) > 8:
                    print(new_reading.Fx - sensor_reading.Fx, new_reading.Fy - sensor_reading.Fy,
                          new_reading.Fz - sensor_reading.Fz)
                    # open gripper
                    open_gripper(gripper)
                    break

            name_changed = False
            pose_changed = False
            time.sleep(2)
            robot.go_home()

