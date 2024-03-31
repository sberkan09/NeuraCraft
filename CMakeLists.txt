cmake_minimum_required(VERSION 3.10)

project(NeuraCraft)

include_directories(.)

# add_library(NeuraCraft SHARED Layer.cpp Matrix.cpp)

add_executable(NeuraCraft Main.cpp Matrix.cpp Layer.cpp)

set_target_properties(NeuraCraft PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
