#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "turbo::tann" for configuration ""
set_property(TARGET turbo::tann APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(turbo::tann PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libtann.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS turbo::tann )
list(APPEND _IMPORT_CHECK_FILES_FOR_turbo::tann "${_IMPORT_PREFIX}/lib/libtann.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
