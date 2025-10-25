cmake_minimum_required(VERSION 3.10)
project(SPH_Fluid_Simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for parallel computing performance
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /openmp")
endif()

# Find required packages
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${GLM_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/glad.c
)

# Create executable
add_executable(sph_simulation ${SOURCES})

# Link libraries
target_link_libraries(sph_simulation
    glfw
    ${OPENGL_LIBRARIES}
    ${CMAKE_DL_LIBS}
    Threads::Threads
)

# Platform-specific settings
if(UNIX AND NOT APPLE)
    target_link_libraries(sph_simulation X11 Xrandr Xi)
elseif(APPLE)
    target_link_libraries(sph_simulation "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
endif()

# Copy shaders to build directory (if you add external shader files later)
# file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})

# Install target
install(TARGETS sph_simulation DESTINATION bin)

# Print configuration info
message(STATUS "")
message(STATUS "========== SPH Fluid Simulation Configuration ==========")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Thread library: ${CMAKE_THREAD_LIBS_INIT}")
message(STATUS "Hardware concurrency: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "========================================================")
message(STATUS "")