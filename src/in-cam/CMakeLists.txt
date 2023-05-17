cmake_minimum_required(VERSION 3.0.2)
project(in-cam)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

find_package(Qt5 REQUIRED Core Gui Widgets REQUIRED)
set(QT_LIBRARIES Qt5::Widgets)

#find_package(yaml-cpp REQUIRED)

find_package(OpenCV REQUIRED)

#find_package( apriltag REQUIRED)

catkin_package()

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${apriltag_INCLUDE_DIRS}
)



file(GLOB QT_FORMS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.ui)
file(GLOB QT_RESOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.qrc)
file(GLOB_RECURSE QT_MOC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS src/*.hpp *.h)
file(GLOB_RECURSE QT_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS src/*.cpp *.cc)

QT5_ADD_RESOURCES(QT_RESOURCES_CPP ${QT_RESOURCES})
QT5_WRAP_UI(QT_FORMS_HPP ${QT_FORMS})
QT5_WRAP_CPP(QT_MOC_HPP ${QT_MOC})

add_executable(calib ${QT_SOURCES} ${QT_RESOURCES_CPP} ${QT_FORMS_HPP} ${QT_MOC_HPP})
target_link_libraries(calib
    ${QT_LIBRARIES}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
#    ${apriltag_LIBS}
#    yaml-cpp
    )
