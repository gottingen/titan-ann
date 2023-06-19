
find_path(BLURBIRD_INCLUDE_PATH NAMES bluebird/bits/bitmap.h)
find_library(BLURBIRD_LIB NAMES libbits.a bits)
include_directories(${BLURBIRD_INCLUDE_PATH})
if((NOT BLURBIRD_INCLUDE_PATH) OR (NOT BLURBIRD_LIB))
    message(FATAL_ERROR "Fail to find bluebird")
endif()
