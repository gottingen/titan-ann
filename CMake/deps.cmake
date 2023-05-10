set(CPM_DOWNLOAD_VERSION 0.32.3)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake...")
    file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
    NAME openmp
    GITHUB_REPOSITORY "gottingen/openmp"
    GIT_TAG 11daa2021c590dc74a0e734b4783570b619d88c9
    EXCLUDE_FROM_ALL YES
    OPTIONS "CMAKE_BUILD_TYPE Release"
            "OPENMP_ENABLE_LIBOMPTARGET OFF"
            "OPENMP_STANDALONE_BUILD ON")

CPMAddPackage(
        NAME turbo
        GITHUB_REPOSITORY "gottingen/turbo"
        GIT_TAG v0.8.6
        OPTIONS "CMAKE_BUILD_TYPE Release"
        "TURBO_BUILD_TESTING OFF"
        "TURBO_BUILD_EXAMPLE OFF")