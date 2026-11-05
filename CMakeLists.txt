cmake_minimum_required(VERSION 3.16)

project(NeuraCraft VERSION 1.1.0 LANGUAGES CXX)

option(NEURACRAFT_BUILD_TESTS "Build the unit tests" ON)
option(NEURACRAFT_USE_ACCELERATE "Use Apple Accelerate for CPU matrix products (macOS)" ON)
option(NEURACRAFT_USE_METAL "Build the Metal GPU backend (macOS)" ON)
option(NEURACRAFT_NATIVE "Optimize for the build machine's CPU (-march=native / -mcpu=native)" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

# ---- Library ------------------------------------------------------------------
# Target is neuracraft_lib (not "neuracraft") so it cannot collide with the
# NeuraCraft executable on case-insensitive file systems (macOS default).
add_library(neuracraft_lib STATIC
    src/Activations.cpp
    src/Backend.cpp
    src/Conv.cpp
    src/Data.cpp
    src/Dense.cpp
    src/Depth.cpp
    src/Diffusion.cpp
    src/Image.cpp
    src/Interpolate.cpp
    src/Json.cpp
    src/Layer.cpp
    src/Lora.cpp
    src/Loss.cpp
    src/Matrix.cpp
    src/Network.cpp
    src/Optimizer.cpp
    src/Parallel.cpp
    src/Pretrained.cpp
    src/Quantized.cpp
    src/Random.cpp
    src/SafeTensors.cpp
    src/Text.cpp
    src/Transformer.cpp
    src/UNet.cpp
    src/Upscale.cpp
    src/UnicodeTables.cpp
)
set_target_properties(neuracraft_lib PROPERTIES OUTPUT_NAME neuracraft)
target_include_directories(neuracraft_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Includes)
target_link_libraries(neuracraft_lib PUBLIC Threads::Threads)

if(MSVC)
    target_compile_options(neuracraft_lib PRIVATE /W4)
else()
    target_compile_options(neuracraft_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    if(NEURACRAFT_NATIVE)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-march=native NC_HAS_MARCH_NATIVE)
        check_cxx_compiler_flag(-mcpu=native NC_HAS_MCPU_NATIVE)
        if(NC_HAS_MARCH_NATIVE)
            target_compile_options(neuracraft_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
        elseif(NC_HAS_MCPU_NATIVE)
            target_compile_options(neuracraft_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mcpu=native>)
        endif()
    endif()
endif()

if(APPLE AND NEURACRAFT_USE_ACCELERATE)
    target_compile_definitions(neuracraft_lib PRIVATE NEURACRAFT_HAS_ACCELERATE)
    target_link_libraries(neuracraft_lib PUBLIC "-framework Accelerate")
endif()

if(APPLE AND NEURACRAFT_USE_METAL)
    enable_language(OBJCXX)
    target_sources(neuracraft_lib PRIVATE src/MetalBackend.mm)
    set_source_files_properties(src/MetalBackend.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_compile_definitions(neuracraft_lib PRIVATE NEURACRAFT_HAS_METAL)
    target_link_libraries(neuracraft_lib PUBLIC "-framework Metal" "-framework Foundation")
endif()

# ---- Demo executable (bin/NeuraCraft) ----------------------------------------------
add_executable(NeuraCraft src/Main.cpp)
target_link_libraries(NeuraCraft PRIVATE neuracraft_lib)
set_target_properties(NeuraCraft PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# ---- Tests ----------------------------------------------------------------------------
if(NEURACRAFT_BUILD_TESTS)
    enable_testing()
    add_executable(neuracraft_tests tests/tests.cpp)
    target_link_libraries(neuracraft_tests PRIVATE neuracraft_lib)
    add_test(NAME unit_tests COMMAND neuracraft_tests)
    add_test(NAME demo_xor COMMAND NeuraCraft xor)
    add_test(NAME demo_spiral COMMAND NeuraCraft spiral)
    add_test(NAME demo_sine COMMAND NeuraCraft sine)
    add_test(NAME bench_quick COMMAND NeuraCraft bench --quick)
endif()
