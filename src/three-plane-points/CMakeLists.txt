cmake_minimum_required(VERSION 3.5.0)
project(three-plane-points VERSION 0.1.0 LANGUAGES C CXX)


find_package(PCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options)



add_executable(three-plane-points main.cpp)

