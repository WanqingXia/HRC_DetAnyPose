cmake_minimum_required(VERSION 3.5.1)
project(workbench_description)

find_package(catkin REQUIRED COMPONENTS
  roslaunch
)

###################################
## catkin specific configuration ##
###################################
catkin_package()

#############
## Install ##
#############

# Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY
  launch
  urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

