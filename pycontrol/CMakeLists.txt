cmake_minimum_required(VERSION 3.0.2)
project(pycontrol)
## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES pycontrol
#  CATKIN_DEPENDS rospy std_msgs
#  DEPENDS system_lib
)
catkin_package()

catkin_install_python(PROGRAMS scripts/media.py scripts/pose.py
  DESTINATION $(CATKIN_PACKAGE_BIN_DESTINATION)
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Install ## 
install(DIRECTORY scripts src
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

