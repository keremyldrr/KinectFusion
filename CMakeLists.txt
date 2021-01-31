cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE STRING INTERNAL FORCE)

project(kinectfusion)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenCV 4.5.1 EXACT REQUIRED)

find_package(PkgConfig REQUIRED)
find_library(LZ4_LIBRARY NAMES lz4)
pkg_check_modules(FLANN REQUIRED flann)

find_package(CUDA 11.0 REQUIRED)
IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_52,code=sm_52)
ENDIF ()

# add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

# Set files to be compiled
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
file(GLOB HEADER_FILES ${PROJECT_INCLUDE_DIR}/*.h)
file(GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

include_directories(${PROJECT_INCLUDE_DIR})
link_directories(${FLANN_LIBRARY_DIRS}) #${FreeImage_LIBRARY_DIR}

cuda_add_executable(${PROJECT_NAME} ${SOURCE_FILES}) # ${HEADER_FILES}
target_link_libraries(${PROJECT_NAME} lz4 ceres freeimage Eigen3::Eigen flann ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC ${Eigen3_INCLUDE_DIR} ${Ceres_INCLUDE_DIR} ${FLANN_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS}) #${FreeImage_INCLUDE_DIR}

