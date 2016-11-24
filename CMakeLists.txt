cmake_minimum_required(VERSION 2.6)
set(TARGET planham)
set(CMAKE_CXX_FLAGS "-Wall -pedantic -g -std=c++0x -O2")

set(SOURCE
  ${PROJECT_SOURCE_DIR}/planham.cpp
)

add_executable(${TARGET} ${SOURCE})
target_link_libraries(${TARGET})