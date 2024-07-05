cmake_minimum_required(VERSION 3.0.2)
project(vention_conveyor_driver)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS rospy std_msgs
)

catkin_install_python(PROGRAMS bin/conveyor_launch.py
  DESTINATION $(CATKIN_PACKAGE_BIN_DESTINATION)
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
## Install ## 
install(DIRECTORY src bin
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
