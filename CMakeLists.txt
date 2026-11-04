cmake_minimum_required(VERSION 3.10)

project(NeuraCraft VERSION 1.0.0 LANGUAGES CXX)

option(NEURACRAFT_BUILD_TESTS "Build the unit tests" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Library ------------------------------------------------------------------
add_library(neuracraft STATIC
    src/Activations.cpp
    src/Data.cpp
    src/Dense.cpp
    src/Loss.cpp
    src/Matrix.cpp
    src/Network.cpp
    src/Optimizer.cpp
    src/Random.cpp
)
target_include_directories(neuracraft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Includes)
if(MSVC)
    target_compile_options(neuracraft PRIVATE /W4)
else()
    target_compile_options(neuracraft PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- Demo executable (bin/NeuraCraft) ----------------------------------------------
add_executable(NeuraCraft src/Main.cpp)
target_link_libraries(NeuraCraft PRIVATE neuracraft)
set_target_properties(NeuraCraft PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# ---- Tests ----------------------------------------------------------------------------
if(NEURACRAFT_BUILD_TESTS)
    enable_testing()
    add_executable(neuracraft_tests tests/tests.cpp)
    target_link_libraries(neuracraft_tests PRIVATE neuracraft)
    add_test(NAME unit_tests COMMAND neuracraft_tests)
    add_test(NAME demo_xor COMMAND NeuraCraft xor)
    add_test(NAME demo_spiral COMMAND NeuraCraft spiral)
    add_test(NAME demo_sine COMMAND NeuraCraft sine)
endif()
