cmake_minimum_required(VERSION 3.0.2)
project(vention_conveyor_msgs)


find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)


## Generate messages in the 'msg' folder
add_message_files(DIRECTORY msg FILES
  ConveyorStat.msg
  ConveyorCmd.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
## Install ## 
install(DIRECTORY msg
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
