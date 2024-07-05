cmake_minimum_required(VERSION 3.0.2)
project(vitpose_run)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
)

# catkin_python_setup()

catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS rospy std_msgs geometry_msgs
)


catkin_install_python(PROGRAMS src/vision_ros.py
  DESTINATION $(CATKIN_PACKAGE_BIN_DESTINATION)
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Install ## 
install(DIRECTORY src
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
