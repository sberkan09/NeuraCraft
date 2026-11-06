cmake_minimum_required(VERSION 3.16)

project(NeuraCraft VERSION 1.1.0 LANGUAGES CXX)

option(NEURACRAFT_BUILD_TESTS "Build the unit tests" ON)
option(NEURACRAFT_USE_ACCELERATE "Use Apple Accelerate for CPU matrix products (macOS)" ON)
option(NEURACRAFT_USE_METAL "Build the Metal GPU backend (macOS)" ON)
option(NEURACRAFT_USE_CUDA "Build the CUDA GPU backend (NVIDIA; needs the CUDA toolkit)" OFF)
option(NEURACRAFT_NATIVE "Optimize for the build machine's CPU (-march=native / -mcpu=native)" OFF)
option(NEURACRAFT_FUZZ "Build the libFuzzer targets in fuzz/ (Clang; the library gets ASan + UBSan)" OFF)

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
    src/Checkpoint.cpp
    src/Onnx.cpp
    src/OnnxOps.cpp
    src/OnnxPipeline.cpp
    src/Conv.cpp
    src/Data.cpp
    src/Dense.cpp
    src/Depth.cpp
    src/Diffusion.cpp
    src/GpuGraph.cpp
    src/Image.cpp
    src/Interpolate.cpp
    src/IpAdapter.cpp
    src/Json.cpp
    src/Layer.cpp
    src/Lora.cpp
    src/Loss.cpp
    src/Matrix.cpp
    src/ModelCache.cpp
    src/Network.cpp
    src/Optimizer.cpp
    src/Parallel.cpp
    src/Pretrained.cpp
    src/Profile.cpp
    src/Quantized.cpp
    src/Random.cpp
    src/SafeTensors.cpp
    src/Server.cpp
    src/Text.cpp
    src/Transformer.cpp
    src/UNet.cpp
    src/Upscale.cpp
    src/UnicodeTables.cpp
    src/VectorMath.cpp
    src/Workflow.cpp
    src/WorkflowSchema.cpp
)
set_target_properties(neuracraft_lib PROPERTIES OUTPUT_NAME neuracraft)
target_include_directories(neuracraft_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Includes)
target_link_libraries(neuracraft_lib PUBLIC Threads::Threads)

if(MSVC)
    target_compile_options(neuracraft_lib PRIVATE /W4)
else()
    target_compile_options(neuracraft_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    # The exp / erf loops clamp with min / max; without trapping-math semantics GCC vectorizes
    # them with plain SSE2 too (nothing in NeuraCraft reads floating-point exception flags).
    set_source_files_properties(src/VectorMath.cpp PROPERTIES COMPILE_OPTIONS "-fno-trapping-math")
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

if(NEURACRAFT_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)  # Turing .. Hopper
    endif()
    set_target_properties(neuracraft_lib PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_sources(neuracraft_lib PRIVATE src/CudaBackend.cu)
    target_compile_definitions(neuracraft_lib PRIVATE NEURACRAFT_HAS_CUDA)
    target_link_libraries(neuracraft_lib PUBLIC CUDA::cudart CUDA::cublas)
endif()

if(APPLE AND NEURACRAFT_USE_METAL)
    enable_language(OBJCXX)
    target_sources(neuracraft_lib PRIVATE src/MetalBackend.mm)
    set_source_files_properties(src/MetalBackend.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_compile_definitions(neuracraft_lib PRIVATE NEURACRAFT_HAS_METAL)
    target_link_libraries(neuracraft_lib PUBLIC "-framework Metal" "-framework Foundation")
endif()

# ---- Fuzz targets (fuzz/README.md) -----------------------------------------------------
if(NEURACRAFT_FUZZ)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "NEURACRAFT_FUZZ needs Clang (libFuzzer)")
    endif()
    target_compile_options(neuracraft_lib PUBLIC -fsanitize=fuzzer-no-link,address,undefined -fno-sanitize-recover=undefined -g)
    target_link_options(neuracraft_lib PUBLIC -fsanitize=address,undefined)
    foreach(target json workflow png safetensors onnx tensorproto)
        add_executable(fuzz_${target} fuzz/fuzz_${target}.cpp)
        target_link_libraries(fuzz_${target} PRIVATE neuracraft_lib)
        target_link_options(fuzz_${target} PRIVATE -fsanitize=fuzzer)
    endforeach()
    return()  # the demo and the tests are not built with the sanitizers
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
    target_compile_definitions(neuracraft_tests PRIVATE NEURACRAFT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME unit_tests COMMAND neuracraft_tests)
    add_test(NAME demo_xor COMMAND NeuraCraft xor)
    add_test(NAME demo_spiral COMMAND NeuraCraft spiral)
    add_test(NAME demo_sine COMMAND NeuraCraft sine)
    add_test(NAME bench_quick COMMAND NeuraCraft bench --quick)
endif()
