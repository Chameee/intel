cmake_minimum_required(VERSION 3.20)
project(intel)

set(CMAKE_CXX_STANDARD 17)


add_executable(intel main.cpp modules/ReLU.h modules/ConvModule.h modules/BatchNorm.h)
