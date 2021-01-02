cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE STRING INTERNAL FORCE)

project(kinectfusion)

# Use pre-compiled libraries on Windows
if(WIN32)
    set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage/x64/ CACHE PATH "Path to FreeImage header file")
    set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
endif(WIN32)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FLANN REQUIRED flann)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
endif()

# Set files to be compiled
file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)

link_directories(${FreeImage_LIBRARY_DIR} ${FLANN_LIBRARY_DIRS})
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ceres freeimage Eigen3::Eigen flann ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC ${Eigen3_INCLUDE_DIR} ${Ceres_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${FLANN_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} include/)

# Visual Studio properties
# set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
# set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

# On Windows copy dll to executable folder
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD        # Adds a post-build event to MyTest
            COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
            "${LIBRARY_DIR}/FreeImage/x64/FreeImage.dll"      # <--this is in-file
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)                 # <--this is out-file path
endif(WIN32)