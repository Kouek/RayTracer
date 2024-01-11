cmake_minimum_required(VERSION 3.18)

project("RayTracer" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/3rd")
set(INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")

# <dependencies>
include_directories(${THIRD_PARTY_DIR})
# <glfw>
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${THIRD_PARTY_DIR}/glfw")
# </glfw>
# <imgui>
aux_source_directory("${THIRD_PARTY_DIR}/imgui/" IMGUI_SRCS)
aux_source_directory("${THIRD_PARTY_DIR}/implot/" IMPLOT_SRCS)
# </imgui>
# </dependencies>

# <global_includes>
include_directories("${THIRD_PARTY_DIR}/glad/include")
include_directories("${THIRD_PARTY_DIR}/imgui")
include_directories(${INCLUDE_DIR})
# </global_includes>

# <lib>
# <ray_tracer>
set(TARGET_NAME "ray_tracer")
file(GLOB CUDA_SRC "${CMAKE_CURRENT_LIST_DIR}/src/ray_tracer/*.cu")
add_library(
	${TARGET_NAME}
	${CUDA_SRC}
)
target_compile_options(
	${TARGET_NAME}
	PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
target_include_directories(
	${TARGET_NAME}
    PUBLIC
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
# </ray_tracer>
# <ray_caster>
set(TARGET_NAME "ray_caster")
file(GLOB CUDA_SRC "${CMAKE_CURRENT_LIST_DIR}/src/ray_caster/*.cu")
add_library(
	${TARGET_NAME}
	${CUDA_SRC}
)
target_compile_options(
	${TARGET_NAME}
	PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
target_include_directories(
	${TARGET_NAME}
    PUBLIC
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
# </ray_caster>
# </lib>

# <app>
# <build_lbvh>
set(TARGET_NAME "build_lbvh")
add_executable(
	${TARGET_NAME}
	"./app/build_lbvh.cpp"
	"${THIRD_PARTY_DIR}/glad/src/glad.c"
	${IMGUI_SRCS}
	${IMPLOT_SRCS}
)
target_link_libraries(
	${TARGET_NAME}
	PRIVATE
	"glfw"
	"ray_tracer"
)
# </build_lbvh>
# <dvr_dpbx_vdb>
set(TARGET_NAME "dvr_dpbx_vdb")
add_executable(
	${TARGET_NAME}
	"./app/dvr_dpbx_vdb.cpp"
	"${THIRD_PARTY_DIR}/glad/src/glad.c"
	${IMGUI_SRCS}
	${IMPLOT_SRCS}
)
target_link_libraries(
	${TARGET_NAME}
	PRIVATE
	"glfw"
	"ray_caster"
)
# </build_lbvh>
# </app>
