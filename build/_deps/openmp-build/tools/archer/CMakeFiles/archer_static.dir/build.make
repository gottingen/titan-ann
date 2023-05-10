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

# Include any dependencies generated for this target.
include _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/flags.make

_deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/ompt-tsan.cpp.o: _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/flags.make
_deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/ompt-tsan.cpp.o: _deps/openmp-src/tools/archer/ompt-tsan.cpp
_deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/ompt-tsan.cpp.o: _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/github/gottingen/titan-ann/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/ompt-tsan.cpp.o"
	cd /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-build/tools/archer && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/ompt-tsan.cpp.o -MF CMakeFiles/archer_static.dir/ompt-tsan.cpp.o.d -o CMakeFiles/archer_static.dir/ompt-tsan.cpp.o -c /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-src/tools/archer/ompt-tsan.cpp

_deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/ompt-tsan.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/archer_static.dir/ompt-tsan.cpp.i"
	cd /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-build/tools/archer && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-src/tools/archer/ompt-tsan.cpp > CMakeFiles/archer_static.dir/ompt-tsan.cpp.i

_deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/ompt-tsan.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/archer_static.dir/ompt-tsan.cpp.s"
	cd /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-build/tools/archer && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-src/tools/archer/ompt-tsan.cpp -o CMakeFiles/archer_static.dir/ompt-tsan.cpp.s

# Object files for target archer_static
archer_static_OBJECTS = \
"CMakeFiles/archer_static.dir/ompt-tsan.cpp.o"

# External object files for target archer_static
archer_static_EXTERNAL_OBJECTS =

_deps/openmp-build/tools/archer/libarcher_static.a: _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/ompt-tsan.cpp.o
_deps/openmp-build/tools/archer/libarcher_static.a: _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/build.make
_deps/openmp-build/tools/archer/libarcher_static.a: _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/github/gottingen/titan-ann/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libarcher_static.a"
	cd /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-build/tools/archer && $(CMAKE_COMMAND) -P CMakeFiles/archer_static.dir/cmake_clean_target.cmake
	cd /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-build/tools/archer && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/archer_static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/build: _deps/openmp-build/tools/archer/libarcher_static.a
.PHONY : _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/build

_deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/clean:
	cd /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-build/tools/archer && $(CMAKE_COMMAND) -P CMakeFiles/archer_static.dir/cmake_clean.cmake
.PHONY : _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/clean

_deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/depend:
	cd /home/ubuntu/github/gottingen/titan-ann/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/github/gottingen/titan-ann /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-src/tools/archer /home/ubuntu/github/gottingen/titan-ann/build /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-build/tools/archer /home/ubuntu/github/gottingen/titan-ann/build/_deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/openmp-build/tools/archer/CMakeFiles/archer_static.dir/depend

