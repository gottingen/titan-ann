# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/github/gottingen/titan-ann

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/github/gottingen/titan-ann/build

# Utility rule file for check-ompt.

# Include any custom commands dependencies for this target.
include _deps/openmp-build/runtime/test/CMakeFiles/check-ompt.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/openmp-build/runtime/test/CMakeFiles/check-ompt.dir/progress.make

_deps/openmp-build/runtime/test/CMakeFiles/check-ompt:
	cd /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-build/runtime/test && /usr/bin/cmake -E echo check-ompt\ does\ nothing,\ dependencies\ not\ found.

check-ompt: _deps/openmp-build/runtime/test/CMakeFiles/check-ompt
check-ompt: _deps/openmp-build/runtime/test/CMakeFiles/check-ompt.dir/build.make
.PHONY : check-ompt

# Rule to build all files generated by this target.
_deps/openmp-build/runtime/test/CMakeFiles/check-ompt.dir/build: check-ompt
.PHONY : _deps/openmp-build/runtime/test/CMakeFiles/check-ompt.dir/build

_deps/openmp-build/runtime/test/CMakeFiles/check-ompt.dir/clean:
	cd /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-build/runtime/test && $(CMAKE_COMMAND) -P CMakeFiles/check-ompt.dir/cmake_clean.cmake
.PHONY : _deps/openmp-build/runtime/test/CMakeFiles/check-ompt.dir/clean

_deps/openmp-build/runtime/test/CMakeFiles/check-ompt.dir/depend:
	cd /home/ubuntu/github/gottingen/titan-ann/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/github/gottingen/titan-ann /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-src/runtime/test /home/ubuntu/github/gottingen/titan-ann/build /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-build/runtime/test /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-build/runtime/test/CMakeFiles/check-ompt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/openmp-build/runtime/test/CMakeFiles/check-ompt.dir/depend

