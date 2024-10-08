cmake_minimum_required(VERSION 3.0.2)
project(dso_ros)

set(PCL_DIR "/usr/lib/x86_64-linux-gnu/cmake/pcl")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  sensor_msgs
  cv_bridge
  message_filters
  PCL
)

set(DSO_PATH "/home/rpm/catkin_ws/src/lidar-dso")#$ENV{DSO_PATH})
message("\n\n---- LOOKING FOR DSO at\n\"${DSO_PATH}\"")


find_package(Pangolin 0.2 REQUIRED)
message("\n\n---- FOUND Pangolin Headers at\n\"${Pangolin_INCLUDE_DIRS}\"")
message("---- FOUND Pangolin Libs at\n\"${Pangolin_LIBRARIES}\"")

find_package(OpenCV REQUIRED)
message("\n\n---- FOUND OpenCV Headers at\n\"${OpenCV_INCLUDE_DIRS}\"")
message("---- FOUND OpenCV Libs at\n\"${OpenCV_LIBS}\"")

find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread)
find_library(DSO_LIBRARY dso ${DSO_PATH}/build/lib)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    roscpp
    sensor_msgs
    cv_bridge
    message_filters
)


###########
## Build ##
###########

set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O2 -g -std=c++0x -march=native -fno-omit-frame-pointer"
)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${DSO_PATH}/src
  ${DSO_PATH}/thirdparty/Sophus
  ${OpenCV_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PCL_LIBRARY_DIRS}
)


add_definitions(
  ${PCL_DEFINITIONS}
)

## Declare a C++ executable
add_executable(dso_live src/main.cpp)

add_dependencies(dso_live ${catkin_EXPORTED_TARGETS})

target_link_libraries(dso_live
    ${DSO_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES})
