# Martin Konrad <konrad@ikp.tu-darmstadt.de>
# License: GPLv2/v3
#
# Try to find libturbo (Perl Compatible Regular Expressions)
#
# Once done this will define
#
# TURBO_FOUND - system has libturbo
# TURBO_INCLUDE_DIR - the libturbo include directory
# TURBO_LIBRARY - where to find libturbo
# TURBO_LIBRARIES - Link these to use libturbo

if(TURBO_INCLUDE_DIR AND TURBO_LIBRARIES)
    # in cache already
    set(TURBO_FOUND TRUE)
else(TURBO_INCLUDE_DIR AND TURBO_LIBRARIES)
    if(NOT WIN32)
        # use pkg-config to get the directories and then use these values
        # in the FIND_PATH() and FIND_LIBRARY() calls
        find_package(PkgConfig)
        pkg_check_modules(PC_TURBO libturbo)
    endif(NOT WIN32)

    find_path(TURBO_INCLUDE_DIR
        NAMES
        turbo/version.h
        HINTS
        $ENV{CONDA_PREFIX}/include
        PATHS
        ${TURBO_PKG_INCLUDE_DIRS}
        /usr/include
        /usr/local/include
        )

        if (WIN32)
                find_library(TURBO_LIBRARY
                        NAMES
                        libturbo
                        HINTS
                        $ENV{CONDA_PREFIX}/lib
                        PATHS
                        ${TURBO_PKG_LIBRARY_DIRS}
                        ${CMAKE_PREFIX_PATH}
                        ${TURBO_PKG_ROOT}/lib
                        )
        else (WIN32)
                find_library(TURBO_LIBRARY
                        NAMES
                        turbo
                        HINTS
                        $ENV{CONDA_PREFIX}/lib
                        PATHS
                        /usr/lib
                        /usr/local/lib
                        ${TURBO_PKG_LIBRARY_DIRS}
                        )
        endif (WIN32)
    set(TURBO_LIBRARIES ${TURBO_LIBRARY})

    # handle the QUIETLY AND REQUIRED arguments AND set TURBO_FOUND to TRUE if
    # all listed variables are TRUE
    # include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(TURBO DEFAULT_MSG TURBO_LIBRARY TURBO_INCLUDE_DIR)

    mark_as_advanced(TURBO_INCLUDE_DIR TURBO_LIBRARY)
endif(TURBO_INCLUDE_DIR AND TURBO_LIBRARIES)
