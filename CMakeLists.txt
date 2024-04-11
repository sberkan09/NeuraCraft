cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME NeuraCraft)

project(${PROJECT_NAME})

include_directories(./Includes)

file(GLOB SOURCE_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)
