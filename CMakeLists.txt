cmake_minimum_required(VERSION 3.15)
project(zagar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Suppress CMake policy warnings
cmake_policy(SET CMP0148 NEW)
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# Find pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Find Boost (header-only, we just need multiprecision)
# Check common locations on macOS
set(Boost_NO_WARN_NEW_VERSIONS ON)

# Try standard find first
find_package(Boost QUIET)

# If not found, check Homebrew locations
if(NOT Boost_FOUND)
    if(APPLE)
        # Check both ARM and Intel Homebrew paths
        foreach(BREW_PREFIX /opt/homebrew /usr/local)
            if(EXISTS "${BREW_PREFIX}/include/boost")
                set(Boost_INCLUDE_DIRS "${BREW_PREFIX}/include")
                set(Boost_FOUND TRUE)
                message(STATUS "Found Boost headers at ${BREW_PREFIX}/include")
                break()
            endif()
        endforeach()
    endif()
endif()

# If still not found, check for conda environment
if(NOT Boost_FOUND)
    if(DEFINED ENV{CONDA_PREFIX})
        if(EXISTS "$ENV{CONDA_PREFIX}/include/boost")
            set(Boost_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
            set(Boost_FOUND TRUE)
            message(STATUS "Found Boost headers in conda environment")
        endif()
    endif()
endif()

if(NOT Boost_FOUND)
    message(FATAL_ERROR
        "Boost not found. Please install Boost:\n"
        "  macOS:   brew install boost\n"
        "  Ubuntu:  sudo apt-get install libboost-dev\n"
        "  conda:   conda install -c conda-forge boost\n"
        "Or set Boost_INCLUDE_DIRS manually."
    )
endif()

# Find OpenMP (optional)
# Help CMake find Homebrew's libomp on macOS
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE BREW_RESULT
    )
    if(BREW_RESULT EQUAL 0 AND EXISTS "${LIBOMP_PREFIX}")
        set(OpenMP_ROOT ${LIBOMP_PREFIX})
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
        message(STATUS "Found Homebrew libomp at ${LIBOMP_PREFIX}")
    endif()
endif()

find_package(OpenMP QUIET)

# Source files
set(SOURCES
    src/zagar/_core.cpp
    src/zagar/count_binary.cpp
    src/zagar/dp_transitions.cpp
)

# Create the Python module
pybind11_add_module(_core MODULE ${SOURCES})

# Include directories
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/zagar
    ${Boost_INCLUDE_DIRS}
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_core PRIVATE ZAGAR_HAS_OPENMP)
    message(STATUS "OpenMP found - parallel support enabled")
else()
    message(STATUS "OpenMP not found - parallel support disabled (will use serial fallback)")
endif()

# Compiler optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(_core PRIVATE -O3)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(_core PRIVATE /O2)
endif()

# Install the compiled extension into the zagar package.
install(TARGETS _core DESTINATION zagar)

# The pure-Python modules are NOT installed from here.  scikit-build-core's
# wheel.packages = ["src/zagar"] already ships every .py file in the package,
# and installing __init__.py from CMake as well created a real
# site-packages/zagar/ directory that shadowed the editable redirect: an
# editable install served __init__.py from the stale copy while serving
# asymptotics.py and exact_dp.py from src/, so edits to __init__.py silently
# had no effect until the next pip install.  Adding the new modules to this
# list would reintroduce the same trap, so the list is gone instead.
