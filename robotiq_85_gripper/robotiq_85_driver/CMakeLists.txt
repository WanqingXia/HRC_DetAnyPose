cmake_minimum_required(VERSION 3.0.2)
project(robotiq_85_driver)

find_package(catkin REQUIRED COMPONENTS roslaunch rospy)

## Uncomment if the package has a setup.py
catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS bin/robotiq_85_driver bin/robotiq_85_test 
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
                      
