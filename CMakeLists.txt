cmake_minimum_required(VERSION 3.18)

project("RayTracer" LANGUAGES CXX C CUDA)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/3rd")
set(INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")

configure_file(
	"${INCLUDE_DIR}/cmake_in"
	"${INCLUDE_DIR}/cmake_in.h"
)

# <dependencies>
add_subdirectory("${THIRD_PARTY_DIR}/glfw")
# </dependencies>

# <global_includes>
include_directories(${THIRD_PARTY_DIR})
include_directories("${THIRD_PARTY_DIR}/glad/include")
include_directories("${THIRD_PARTY_DIR}/glm")
include_directories(${INCLUDE_DIR})
# </global_includes>

# <lib>
# <ray_tracer>
set(TARGET_NAME "ray_tracer")
file(GLOB SRC "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
file(GLOB CUDA_SRC "${CMAKE_CURRENT_LIST_DIR}/src/*.cu")
file(GLOB CUDA_KN "${CMAKE_CURRENT_LIST_DIR}/kernel/*.cu")
add_library(
	${TARGET_NAME}
	${SRC} ${CUDA_SRC} ${CUDA_KN}
	"${THIRD_PARTY_DIR}/glad/src/glad.c"
)
target_compile_options(
	${TARGET_NAME}
	PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
target_include_directories(
	${TARGET_NAME}
	PRIVATE
	"${CMAKE_CURRENT_LIST_DIR}/kernel"
    PUBLIC
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
)
# </ray_tracer>
# </lib>

# <app>
set(TARGET_NAME "demo")
add_executable(
	${TARGET_NAME}
	"main.cpp"
	"${THIRD_PARTY_DIR}/glad/src/glad.c"
	"${THIRD_PARTY_DIR}/tinyxml/tinyxml2.cpp"
)
target_link_libraries(
	${TARGET_NAME}
	PRIVATE
	"ray_tracer"
	"glfw"
)
# </app>
