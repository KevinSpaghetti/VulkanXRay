cmake_minimum_required(VERSION 3.23)
project(VulkanXRay)

set(CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(libs/glfw)

SET(IMGUI_FILES
    libs/imgui/imgui.cpp
    libs/imgui/imgui_draw.cpp
    libs/imgui/imgui_widgets.cpp
    libs/imgui/imgui_demo.cpp
    libs/imgui/imgui_tables.cpp
    libs/imgui/backends/imgui_impl_glfw.cpp
    libs/imgui/backends/imgui_impl_vulkan.cpp)

add_executable(VulkanXRay main.cpp ${IMGUI_FILES})

target_include_directories(VulkanXRay PUBLIC libs/glfw/include)
target_include_directories(VulkanXRay PUBLIC libs/imgui)
target_include_directories(VulkanXRay PUBLIC libs/imgui/backends)
target_include_directories(VulkanXRay PUBLIC $ENV{VULKAN_SDK}/Include)

target_link_directories(VulkanXRay PUBLIC libs/glfw/lib)
target_link_libraries(VulkanXRay PUBLIC ${Vulkan_LIBRARIES} glfw)

add_custom_command(TARGET VulkanXRay POST_BUILD
        COMMENT "Compiling shaders into sprv"
        COMMAND glslc ${CMAKE_SOURCE_DIR}/shaders/vertex.vert -o vertex.sprv
        COMMAND glslc ${CMAKE_SOURCE_DIR}/shaders/fragment.frag -o fragment.sprv
        )