cmake_minimum_required(VERSION 3.10)

project(NeuraCraft)

include_directories(./Includes)

set(SOURCE_FILES
    Main.cpp
    Matrix.cpp
    Layer.cpp
)

# add_library(NeuraCraft SHARED ${SOURCE_FILES})

add_executable(NeuraCraft ${SOURCE_FILES})

set_target_properties(NeuraCraft PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
