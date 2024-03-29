cmake_minimum_required(VERSION 3.15)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE STRING INTERNAL FORCE)

project(kinectfusion CUDA CXX C)

# Set C++ flags
set(CMAKE_CUDA_FLAGS "-fopenmp --default-stream per-thread -O3 -lcudpp -Xcompiler -mavx -Xcompiler -mavx2 -Xcompiler -O3 -Xcompiler -fopenmp -w --disable-warnings")
set(CUDA_DIR "CUDA PATH")
set(CUDA_LIBRARY_DIR "CUDA LIBRARY PATH" )
set(CMAKE_CUDA_ARCHITECTURES 60)
find_package(CUDA 10.0 REQUIRED)
IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11)
ENDIF ()
find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}/home/antares/kyildiri/anaconda3/envs/KinectFusion/include/ceres
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenCV 4.5.0 EXACT REQUIRED PATHS OPENCV GPU PATH)

find_package(PkgConfig REQUIRED)
find_library(LZ4_LIBRARY NAMES lz4)
set(FLANN_LIBRARY_DIRS FLANN PATH)
include_directories(Flann)
#pkg_check_modules(FLANN REQUIRED flann)

# add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

# Set files to be compiled
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(PROJECT_KERNEL_DIR ${PROJECT_SOURCE_DIR}/kernels)
file(GLOB HEADER_FILES ${PROJECT_INCLUDE_DIR}/*.h)
file(GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB KERNELS_FILES ${PROJECT_KERNEL_DIR}/*.cu)
set(PROJECT_KERNEL_HEADERS ${PROJECT_KERNEL_DIR}/include/)


link_directories(${FLANN_LIBRARY_DIRS}) #${FreeImage_LIBRARY_DIR}

cuda_add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${KERNELS_FILES}) # ${HEADER_FILES}
target_link_libraries(${PROJECT_NAME} lz4 ceres freeimage Eigen3::Eigen ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC ${Eigen3_INCLUDE_DIR} ${Ceres_INCLUDE_DIR} ${FLANN_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${PROJECT_KERNEL_DIR} ${PROJECT_KERNEL_HEADERS} ${PROJECT_INCLUDE_DIR})

