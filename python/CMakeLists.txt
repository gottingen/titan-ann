# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
cmake_minimum_required(VERSION 3.20)

project(tannpy CXX)

set(CMAKE_CXX_STANDARD 17)

if (PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE _numpy_include
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)


find_package(MKL REQUIRED)
find_package(OpenMP REQUIRED)
#find_package(AIO REQUIRED)
set(CARBIN_SYSTEM_DYLINK)
if (APPLE)
    find_library(CoreFoundation CoreFoundation)
    list(APPEND CARBIN_SYSTEM_DYLINK ${CoreFoundation} pthread)
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND CARBIN_SYSTEM_DYLINK rt dl pthread)
endif ()
find_path(TURBO_INCLUDE_PATH NAMES turbo/version.h)
find_library(TURBO_LIB NAMES turbo)
include_directories(${TURBO_INCLUDE_PATH})
if((NOT TURBO_INCLUDE_PATH) OR (NOT TURBO_LIB))
    message(FATAL_ERROR "Fail to find turbo")
endif()

find_package(tann REQUIRED)
set(CARBIN_DEPS_LINK
        ${TURBO_LIB}
        ${CARBIN_SYSTEM_DYLINK}
        ${MKL_LIBRARIES}
        aio
        OpenMP::OpenMP_CXX
        )
# pybind11_add_module(tannpy MODULE src/tann_bindings.cpp)
# the following is fairly synonymous with pybind11_add_module, but we need more target_link_libraries
# see https://pybind11.readthedocs.io/en/latest/compiling.html#advanced-interface-library-targets for more details
add_library(_tannpy MODULE tannpy/tann_bindings.cc)

if (MSVC)
    target_compile_options(_tannpy PRIVATE /U_WINDLL)
endif()

target_link_libraries(
        _tannpy
        PRIVATE
        pybind11::module
        pybind11::lto
        pybind11::windows_extras
        tann::tann
        ${CARBIN_DEPS_LINK}
)

pybind11_extension(_tannpy)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(_tannpy)
endif()

set_target_properties(_tannpy PROPERTIES CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden")

# generally, the VERSION_INFO flag is set by pyproject.toml, by way of setup.py.
# attempts to locate the version within CMake fail because the version has to be available
# to pyproject.toml for the sdist to work after we build it.

if(NOT VERSION_INFO)
    set(VERSION_INFO "0.0.0dev")
endif()
target_compile_definitions(_tannpy PRIVATE VERSION_INFO="${VERSION_INFO}")
