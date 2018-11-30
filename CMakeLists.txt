cmake_minimum_required(VERSION 3.12)
project(SimpleCNN)

set(CMAKE_CXX_STANDARD 11)

add_executable(SimpleCNN main.cpp utils.h utils.cpp conv_layer.cpp conv_layer.h max_pooling_layer.cpp max_pooling_layer.h fully_connected_layer.cpp fully_connected_layer.h session.cpp session.h)

include_directories(./eigen)