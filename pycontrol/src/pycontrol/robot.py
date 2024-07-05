#!/usr/bin/env python3
#!\file
#
# \author  Wanqing Xia wxia612@aucklanduni.ac.nz
# \date    2022-07-28
#
#
# ---------------------------------------------------------------------

import sys
from turtle import pos
import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
import time
import rospy
import actionlib

from controller_manager_msgs.srv import SwitchControllerRequest, SwitchController
from controller_manager_msgs.srv import LoadControllerRequest, LoadController
import geometry_msgs.msg as geometry_msgs
from tf2_msgs.msg import TFMessage
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, FollowJointTrajectoryFeedback
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from cartesian_control_msgs.msg import (
    FollowCartesianTrajectoryAction,
    FollowCartesianTrajectoryGoal,
    CartesianTrajectoryPoint,
    FollowCartesianTrajectoryActionFeedback,
)

from scipy.spatial.transform import Rotation
from ur_ikfast import ur_kinematics

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# All of those controllers can be used to execute joint-based trajectories.
# The scaled versions should be preferred over the non-scaled versions.
JOINT_TRAJECTORY_CONTROLLERS = [
    "scaled_pos_joint_traj_controller",
    "scaled_vel_joint_traj_controller",
    "pos_joint_traj_controller",
    "vel_joint_traj_controller",
    "forward_joint_traj_controller",
]

# All of those controllers can be used to execute Cartesian trajectories.
# The scaled versions should be preferred over the non-scaled versions.
CARTESIAN_TRAJECTORY_CONTROLLERS = [
    "pose_based_cartesian_traj_controller",
    "joint_based_cartesian_traj_controller",
    "forward_cartesian_traj_controller",
]

# We'll have to make sure that none of these controllers are running, as they will
# be conflicting with the joint trajectory controllers
CONFLICTING_CONTROLLERS = ["joint_group_vel_controller", "twist_controller"]


class UR5eRobot:
    """Small trajectory client to test a joint trajectory"""

    def __init__(self):

        timeout = rospy.Duration(5)
        self.switch_srv = rospy.ServiceProxy(
            "controller_manager/switch_controller", SwitchController
        )
        self.load_srv = rospy.ServiceProxy("controller_manager/load_controller", LoadController)
        try:
            self.switch_srv.wait_for_service(timeout.to_sec())
        except rospy.exceptions.ROSException as err:
            rospy.logerr("Could not reach controller switch service. Msg: {}".format(err))
            sys.exit(-1)

        self.cartesian_trajectory_controller = CARTESIAN_TRAJECTORY_CONTROLLERS[2]
        self.joint_trajectory_controller = JOINT_TRAJECTORY_CONTROLLERS[0]
        # switch to cartesian controller and close al other controllers
        self._switch_controller(self.cartesian_trajectory_controller)
        self._current_controller = self.cartesian_trajectory_controller

        self._ee_pose = geometry_msgs.Pose()
        self._joint_pose = JointTrajectoryPoint()
        self.ur5e_arm = ur_kinematics.URKinematics('ur5e')

        rospy.Subscriber("/tf", TFMessage, self._update_robot_pose, queue_size=10)
        rospy.Subscriber("/joint_states", JointState, self._update_joint_pose, queue_size=10)
        time.sleep(1)

        self.home_pose = np.array([0.0, -50, -130, -55, 90, 0.0])
        # self.go_home()
        
    def go_home(self):
        self.execute_joint_trajectory(self.home_pose)

    def _update_robot_pose(self, data):
        self.last_pose = self._ee_pose
        if data.transforms[0].transform.translation.y == 0 and data.transforms[0].transform.translation.z == 0:
            self._ee_pose = self.last_pose
        else:
            self._ee_pose.position.x = data.transforms[0].transform.translation.x
            self._ee_pose.position.y = data.transforms[0].transform.translation.y
            self._ee_pose.position.z = data.transforms[0].transform.translation.z
            self._ee_pose.orientation.x = data.transforms[0].transform.rotation.x
            self._ee_pose.orientation.y = data.transforms[0].transform.rotation.y
            self._ee_pose.orientation.z = data.transforms[0].transform.rotation.z
            self._ee_pose.orientation.w = data.transforms[0].transform.rotation.w

    def _update_joint_pose(self, data):
        # data from joint_state is in the sequence of "elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", need to be converted
        self._joint_pose.positions = [data.position[2], data.position[1], data.position[0], data.position[3], data.position[4], data.position[5]]

    def get_ee_pose(self):
        position = self._ee_pose.position
        orientation = self._ee_pose.orientation
        ee_pose = np.array([position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w])
        return ee_pose

    def get_joint_pose(self):
        return np.array(self._joint_pose.positions)


    def execute_joint_trajectory(self, joint_pose):
        """Creates a trajectory and sends it using the selected action server"""
        if self._current_controller != self.joint_trajectory_controller:
            self._switch_controller(self.joint_trajectory_controller)
        
        if joint_pose.shape[0] == 6:
            joint_pose = np.deg2rad(joint_pose)

            trajectory_client = actionlib.SimpleActionClient(
                "{}/follow_joint_trajectory".format(self.joint_trajectory_controller),
                FollowJointTrajectoryAction,
            )
            # Wait for action server to be ready
            timeout = rospy.Duration(5)
            if not trajectory_client.wait_for_server(timeout):
                rospy.logerr("Could not reach controller action server.")
                sys.exit(-1)

            # Create and fill trajectory goal
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = JOINT_NAMES

            point = JointTrajectoryPoint()
            point.positions = joint_pose
            time_last = np.max(np.abs(joint_pose - self.get_joint_pose()))
            point.time_from_start = rospy.Duration(time_last)
            goal.trajectory.points.append(point)

            trajectory_client.send_goal(goal)
            trajectory_client.wait_for_result()

            result = trajectory_client.get_result()
            rospy.loginfo("Rotating joints to position {} finished in state {}".format(joint_pose, result.error_code))
        else:
            rospy.logerr("Each action should have 6 elements, this one has {}".format(joint_pose.shape[0]))


    def execute_cartesian_trajectory(self, end_pose:np.array):
        """Creates a Cartesian trajectory and sends it using the selected action server"""
        if self._current_controller != self.cartesian_trajectory_controller:
            self._switch_controller(self.cartesian_trajectory_controller)
        
        rospy.loginfo(f'Moving from current pose {self.get_ee_pose()} to end pose {end_pose}.')
        if end_pose.ndim != 1:
            rospy.logerr("End pose must be a 1D numpy array.")
        if end_pose.shape[0] not in [6, 7]:
            rospy.logerr("End pose must has either 6 (rotvec) or 7 (quaternion) elements.")


        # Determine if the end pose is represented by a rotation vector or quaternion
        if len(end_pose) == 6:
            end_rot = R.from_rotvec(end_pose[3:]).as_quat()
        elif len(end_pose) == 7:
            end_rot = end_pose[3:]
        else:
            raise ValueError("End pose must be either 6 (rotvec) or 7 (quaternion) numbers.")

        # make sure the correct controller is loaded and activated
        goal = FollowCartesianTrajectoryGoal()
        trajectory_client = actionlib.SimpleActionClient(
            "{}/follow_cartesian_trajectory".format(self.cartesian_trajectory_controller),
            FollowCartesianTrajectoryAction,
        )

        # Wait for action server to be ready
        timeout = rospy.Duration(5)
        if not trajectory_client.wait_for_server(timeout):
            rospy.logerr("Could not reach controller action server.")
            sys.exit(-1)


        point = CartesianTrajectoryPoint()
        point.pose = self._convert_pose(end_pose[:3], end_rot)
        point.time_from_start = rospy.Duration(np.linalg.norm(self.get_ee_pose()[:3] - end_pose[:3]) * 3)
        goal.trajectory.points.append(point)

        # rospy.loginfo("Executing trajectory using the {}".format(self.cartesian_trajectory_controller))
        trajectory_client.send_goal(goal)
        trajectory_client.wait_for_result()

        result = trajectory_client.get_result()

        # rospy.loginfo("Moving to point {} finished in state {}".format(end_pose, result.error_code))


    def _switch_controller(self, target_controller):
        """Activates the desired controller and stops all others from the predefined list above"""
        other_controllers = (
            JOINT_TRAJECTORY_CONTROLLERS
            + CARTESIAN_TRAJECTORY_CONTROLLERS
            + CONFLICTING_CONTROLLERS
        )

        other_controllers.remove(target_controller)

        srv = LoadControllerRequest()
        srv.name = target_controller
        self.load_srv(srv)

        srv = SwitchControllerRequest()
        srv.stop_controllers = other_controllers
        srv.start_controllers = [target_controller]
        srv.strictness = SwitchControllerRequest.BEST_EFFORT
        self.switch_srv(srv)
        self._current_controller = target_controller

    def _convert_pose(self, pos, quat):
        return geometry_msgs.Pose(geometry_msgs.Vector3(pos[0], pos[1], pos[2]), geometry_msgs.Quaternion(quat[0], quat[1], quat[2], quat[3]))

    def _vec2quat(self, x, y, z):
        # Create a rotation object from rotation vector in radians
        rot = Rotation.from_rotvec([x, y, z])
        # Convert to quaternions and print
        rq = rot.as_quat()
        return rq[0], rq[1], rq[2], rq[3]

    def _quat2vec(self, x, y, z, w):
        # Create a rotation object from rotation vector in radians
        rot = Rotation.from_quat([x, y, z, w])
        # Convert to quaternions and print
        rr = rot.as_rotvec()
        return rr[0], rr[1], rr[2]
    
    def joint_to_cart(self, joint_pose):
        # convert joint pose(in radians) to cartisian point
        pose_quat = self.ur5e_arm.forward(joint_pose)
        vec = self._quat2vec(pose_quat[3], pose_quat[4], pose_quat[5], pose_quat[6])
        return [pose_quat[0], pose_quat[1], pose_quat[2], vec[1], -vec[0], vec[2]]
        # return [pose_quat[0], pose_quat[1], pose_quat[2], vec[1], -vec[0], vec[2]]
    
    def cart_to_joint(self, cart_pose):
        # convert cartisian pose(in quaternion) to joint pose
        assert len(cart_pose) == 6 or 7, "The cartisian pose must be be a length 6 (rottation vector) or 7 (x, y, z, w format quaternion) numpy array."
        if len(cart_pose) == 6:
            end_rot = R.from_rotvec(cart_pose[3:]).as_quat()
        elif len(cart_pose) == 7:
            end_rot = cart_pose[3:]

        end_rot = np.roll(end_rot, -1)
        pose = np.concatenate((cart_pose[:3], end_rot), axis=0)  # inverse function requires w,x,y,z format quaternion
        joint_pose = self.ur5e_arm.inverse(pose)
        return np.array(joint_pose)

