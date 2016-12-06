cmake_minimum_required(VERSION 2.8)
project(Adaptels)
find_package(OpenCV REQUIRED)

############################### Adaptels executable
add_executable(Adaptels
  examples/adaptels_main.cpp
)
include_directories(include)
target_link_libraries(Adaptels
  ${OpenCV_LIBS}
  AdaptelsLib
)

############################### AdaptelsLib static library
add_library(AdaptelsLib STATIC
  src/adaptels.cpp
)
include_directories(include)
target_link_libraries(AdaptelsLib
  ${OpenCV_LIBS}
)
