cmake_minimum_required(VERSION 3.0.2)
project(robotiq_85_description)

find_package(catkin REQUIRED COMPONENTS roslaunch)

catkin_package()

roslaunch_add_file_check(launch)

install(
  DIRECTORY launch meshes urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
