include("/home/ubuntu/github/gottingen/titan-ann/build/cmake/CPM_0.32.3.cmake")
CPMAddPackage(NAME;turbo;GITHUB_REPOSITORY;gottingen/turbo;GIT_TAG;v0.8.6;OPTIONS;CMAKE_BUILD_TYPE Release;TURBO_BUILD_TESTING OFF;TURBO_BUILD_EXAMPLE OFF)
set(turbo_FOUND TRUE)