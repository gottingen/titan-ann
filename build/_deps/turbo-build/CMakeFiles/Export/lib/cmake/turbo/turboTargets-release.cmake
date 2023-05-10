#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "turbo::turbo" for configuration "Release"
set_property(TARGET turbo::turbo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(turbo::turbo PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libturbo.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS turbo::turbo )
list(APPEND _IMPORT_CHECK_FILES_FOR_turbo::turbo "${_IMPORT_PREFIX}/lib/libturbo.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
