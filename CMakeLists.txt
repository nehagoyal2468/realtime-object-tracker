cmake_minimum_required(VERSION 3.15)
project(realtime_object_tracker VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(tracker
    src/main.cpp
    src/Tracker.cpp
)

target_link_libraries(tracker ${OpenCV_LIBS})