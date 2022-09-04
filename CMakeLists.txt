# Copyright 2022 Wenxin Wang. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file. See the AUTHORS file for names of contributors.

cmake_minimum_required(VERSION 3.8)
project(RadixSort VERSION 1.0 LANGUAGES C CXX)

#####################################
# Add libraries
#####################################

# Adding the Opencl library
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
  message ("OpenCL found")
  include_directories(${OpenCL_INCLUDE_DIRS})
  link_directories(${OpenCL_LIBRARY})
else()
  message (FATAL_ERROR "Cannot find OpenCL")
endif()

# Adding the Boost library
# (Boost library should be installed beforehand)
if (WIN32)
  message("OS: Windows")
  set(BOOST_INC "C:/local/boost_1_76_0_b1_rc2")
  set(BOOST_LIB "C:/local/boost_1_76_0_b1_rc2/lib64-msvc-14.2/")

  include_directories(${BOOST_INC})
  link_directories(${BOOST_LIB})

elseif(UNIX)
  message("OS: Linux")
  include_directories(${Boost_INCLUDE_DIR})

endif()

#compile files in COre and OpenCL external libs
file(GLOB CORE_SRC "${PROJECT_SOURCE_DIR}/include/Core/*.cpp" "${PROJECT_SOURCE_DIR}/include/Core/*.c")
file(GLOB OPENCL_SRC "${PROJECT_SOURCE_DIR}/include/OpenCL/*.cpp" "${PROJECT_SOURCE_DIR}/include/OpenCL/*.c")
file(GLOB BOOST_SRC "${BOOST_LIB}/*.lib")

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Add utils library
add_library(utils_lib STATIC ${PROJECT_SOURCE_DIR}/include/Utils/clFileFinder.cpp)
target_include_directories(utils_lib PRIVATE ${PROJECT_SOURCE_DIR}/include/Utils)


#####################################
# Make executable
#####################################
add_executable (${PROJECT_NAME} "./src/radixSort.cpp" ${CORE_SRC} ${OPENCL_SRC})


#####################################
# Link libraries
#####################################
target_include_directories (${PROJECT_NAME} 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
         ${PROJECT_SOURCE_DIR}/include
         ${PROJECT_SOURCE_DIR}/src)

target_link_libraries (${PROJECT_NAME} 
  ${OpenCL_LIBRARY} dl boost_system utils_lib) #imagehlp )

