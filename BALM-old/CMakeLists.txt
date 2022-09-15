cmake_minimum_required(VERSION 2.8.3)
project(balm)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_conversions
  pcl_ros
  livox_ros_driver
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
	${CERES_INCLUDE_DIRS}
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL OpenCV
  INCLUDE_DIRS
)

add_executable(livox_feature src/features/livox_feature.cpp)
target_link_libraries(livox_feature ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(velodyne_feature src/features/velodyne_feature.cpp)
target_link_libraries(velodyne_feature ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(balm_front_back src/balm_front_back.cpp src/myso3/myso3.hpp src/myso3/myso3.cpp)
target_link_libraries(balm_front_back ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(loamscan2map src/loamscan2map.cpp)
target_link_libraries(loamscan2map ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(balm_only_back src/balm_only_back.cpp src/myso3/myso3.hpp src/myso3/myso3.cpp)
target_link_libraries(balm_only_back ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

