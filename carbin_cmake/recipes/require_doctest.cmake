
find_path(DOCTEST_INCLUDE_PATH NAMES doctest/doctest.h)
include_directories(${DOCTEST_INCLUDE_PATH})
if((NOT DOCTEST_INCLUDE_PATH))
    message(FATAL_ERROR "Fail to find turbo")
endif()
include_directories({DOCTEST_INCLUDE_PATH})