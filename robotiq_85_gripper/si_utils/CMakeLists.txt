cmake_minimum_required(VERSION 3.0.2)
project(si_utils)

## Uncomment if the package has a setup.py
#catkin_python_setup()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roslaunch)

## Declare a catkin package
catkin_package()

roslaunch_add_file_check(launch)

## Install ## 
install(PROGRAMS scripts/timed_roslaunch 
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


