
include(CPM_v0380)

CPMAddPackage(
        NAME turbo
        GITHUB_REPOSITORY "gottingen/turbo"
        GIT_TAG v0.8.6
        OPTIONS "CMAKE_BUILD_TYPE Release"
        "TURBO_BUILD_TESTING OFF"
        "TURBO_BUILD_EXAMPLE OFF")