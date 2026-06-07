cmake_minimum_required(VERSION 3.18)
project(SPLAY VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ────────────────────────────────────────────────────────────
# Options
# ────────────────────────────────────────────────────────────
option(SPLAY_ENABLE_MPI  "Enable MPI parallelism"            ON)
option(SPLAY_ENABLE_TBB  "Enable Intel TBB thread-level par" OFF)
option(SPLAY_BUILD_TESTS "Build unit tests"                  ON)

# ────────────────────────────────────────────────────────────
# Dependencies
# ────────────────────────────────────────────────────────────
include(FetchContent)

# yaml-cpp 0.8.0 uses cmake_minimum_required(2.8.12) which triggers a
# deprecation error in CMake >= 3.27.  Set the policy minimum before
# FetchContent pulls it so the subdirectory configure succeeds.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM "3.5" CACHE STRING
        "Minimum CMake policy version for third-party deps" FORCE)
endif()

# yaml-cpp: fetched automatically; no system install required.
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
    GIT_SHALLOW    TRUE
)
set(YAML_CPP_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL         OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

if(SPLAY_ENABLE_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
endif()

if(SPLAY_ENABLE_TBB)
    find_package(TBB REQUIRED)
    message(STATUS "TBB found")
endif()

# ────────────────────────────────────────────────────────────
# Library: splay_core  (all logic; shared by main + tests)
# ────────────────────────────────────────────────────────────
set(SPLAY_SRCS
    src/config.cpp
    src/gas.cpp
    src/mesh.cpp
    src/state.cpp
    src/transport.cpp
    src/reconstruction.cpp
    src/reconstruction_ppm_pele.cpp
    src/riemann.cpp
    src/viscous_flux.cpp
    src/residual.cpp
    src/rk.cpp
    src/mpi_decomp.cpp
    src/io.cpp
    src/diagnostics.cpp
)

add_library(splay_core STATIC ${SPLAY_SRCS})

target_include_directories(splay_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(splay_core PUBLIC yaml-cpp::yaml-cpp)

if(SPLAY_ENABLE_MPI)
    target_link_libraries(splay_core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(splay_core PUBLIC SPLAY_ENABLE_MPI)
endif()

if(SPLAY_ENABLE_TBB)
    target_link_libraries(splay_core PUBLIC TBB::tbb)
    target_compile_definitions(splay_core PUBLIC SPLAY_ENABLE_TBB)
endif()

# Compiler warnings
target_compile_options(splay_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall -Wextra -Wpedantic>
)

# ────────────────────────────────────────────────────────────
# Executable: splay
# ────────────────────────────────────────────────────────────
add_executable(splay src/main.cpp)
target_link_libraries(splay PRIVATE splay_core)

# ────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────
if(SPLAY_BUILD_TESTS)
    enable_testing()

    set(TEST_SRCS
        tests/test_gas.cpp
        tests/test_transport.cpp
        tests/test_normal_shock.cpp
        tests/test_mpi_decomp.cpp
        tests/test_restart.cpp
        tests/test_sod.cpp
    )

    foreach(test_src ${TEST_SRCS})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE splay_core)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# ────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "SPLAY configuration summary")
message(STATUS "  Version     : ${PROJECT_VERSION}")
message(STATUS "  C++ std     : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  MPI         : ${SPLAY_ENABLE_MPI}")
message(STATUS "  TBB         : ${SPLAY_ENABLE_TBB}")
message(STATUS "  Tests       : ${SPLAY_BUILD_TESTS}")
message(STATUS "  Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "")
