cmake_minimum_required(VERSION 3.12)
project(QHU)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

add_executable(QHU qhu-model.cpp)

target_link_libraries(QHU ${TORCH_LIBRARIES})

set_property(TARGET QHU PROPERTY CXX_STANDARD 14)
