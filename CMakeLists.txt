# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

cmake_minimum_required(VERSION 3.5.1)

project(sampleYOLOV7)

add_compile_options(-std=c++11)

add_definitions(-DENABLE_DVPP_INTERFACE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")





#################################################################pyb
set(MY_PYBIND ${CMAKE_SOURCE_DIR}/pybind11)
add_subdirectory(${MY_PYBIND})
set(DEMO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/sampleYOLOV7.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mat_warper.cpp
)
                
# add demo module
pybind11_add_module(${PROJECT_NAME} ${DEMO_SOURCES})
#################################################################pyb
set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/lastest/runtime/lib64/stub")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    message(STATUS "set LIB_PATH: ${LIB_PATH}")
endif ()

set(THIRDPART $ENV{THIRDPART_PATH})
if (NOT DEFINED ENV{THIRDPART_PATH})
    set(THIRDPART "/usr/local/Ascend/ascend-toolkit/latest/thirdpart")
    message(STATUS "set default THIRDPART: ${THIRDPART}")
else()
    message(STATUS "set THIRDPART: ${THIRDPART}")
endif()


#find opencv
find_package(OpenCV REQUIRED)

#include opencv
include_directories(${OpenCV_INCLUDE_DIRS})

#include 
include_directories(${CMAKE_SOURCE_DIR}/include)

include_directories(
   ${INC_PATH}/runtime/include/
   ${THIRDPART}/include/acllite
)

link_directories(
    ${THIRDPART}/lib/
    ${LIB_PATH}
)
##########################################
# add_executable(${PROJECT_NAME}
#         src/sampleYOLOV7.cpp)
##########################################
#link opencv
target_link_libraries(${PROJECT_NAME}  PRIVATE ${OpenCV_LIBS})

if(target STREQUAL "Simulator_Function")
    target_link_libraries(${PROJECT_NAME} funcsim)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE  stdc++ acllite opencv_core opencv_imgproc opencv_imgcodecs dl rt  )#ascendcl acl_dvpp  ascendcl
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
