cmake_minimum_required(VERSION 3.10)
project(SinusArray)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

option(USE_DOUBLE "Use double instead of float" OFF)

if(USE_DOUBLE)
    add_definitions(-DUSE_DOUBLE)
endif()

add_executable(main main.cpp)
