cmake_minimum_required(VERSION 3.0)
project(assignment1_character)

find_package(OpenGL REQUIRED)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_subdirectory(external)

include_directories(
	external/glfw-3.1.2/include/
	external/glm-0.9.7.1/
	external/glad-opengl-3.3/include/
	external/tinygltf-2.9.3/
	external/
	assignment1/
)

add_executable(assignment1_character
	assignment1/assignment1_character.cpp
	assignment1/render/shader.cpp
)
add_executable(main
	assignment1/main.cpp
	assignment1/render/shader.cpp
)

target_link_libraries(assignment1_character
	${OPENGL_LIBRARY}
	glfw
	glad
)

target_link_libraries(main
	${OPENGL_LIBRARY}
	glfw
	glad
)