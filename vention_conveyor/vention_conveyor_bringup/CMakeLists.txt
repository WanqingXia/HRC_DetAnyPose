cmake_minimum_required(VERSION 3.0.2)
project(vention_conveyor_bringup)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  roslaunch
)

catkin_package()

roslaunch_add_file_check(launch)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
install(DIRECTORY launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
