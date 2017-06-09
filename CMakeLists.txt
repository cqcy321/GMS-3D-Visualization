cmake_minimum_required(VERSION 2.8)
project( gms_match_demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
LIST(APPEND CMAKE_MODULE_PATH ./cmake_modules)
# OpenCV 
find_package( OpenCV REQUIRED )
find_package(Eigen3 3.1.0 REQUIRED)
# 
include_directories( 
 include
${OpenCV_INCLUDE_DIRS} 
${EIGEN3_INCLUDE_DIR}
)


add_executable( gms_match_demo ./src/demo.cpp )
target_link_libraries( gms_match_demo ${OpenCV_LIBS} )
