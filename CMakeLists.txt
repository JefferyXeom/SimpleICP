cmake_minimum_required(VERSION 3.10)

set(CMAKE_TOOLCHAIN_FILE "../vcpkg/scripts/buildsystems/vcpkg.cmake")
message(STATUS "Toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

if (MSVC)
	# 添加编译选项, 设置 输入代码的编码格式为 utf-8
	add_compile_options("/source-charset:utf-8")
endif()

# 项目信息
project(ICPProject LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加项目头文件路径
set(INCLUDE "./include")
include_directories(${INCLUDE})

# 查找 PCL 包
find_package(PCL CONFIG REQUIRED)
message(STATUS "PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS}")
message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")

# 添加 PCL 的头文件和库路径
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

# 查找 Ceres 包
find_package(Ceres CONFIG REQUIRED)
# 输出 Ceres::Ceres 目标的属性（包括包含目录和库路径）
# get_target_property(CERES_INCLUDE_DIRS Ceres::ceres INTERFACE_INCLUDE_DIRECTORIES)
# get_target_property(CERES_LIBRARIES Ceres::ceres INTERFACE_LINK_LIBRARIES)
# message(STATUS "Found Ceres: ${CERES_FOUND}")
# message(STATUS "Ceres Include Dirs: ${CERES_INCLUDE_DIRS}")
# message(STATUS "Ceres LIBRARIES: ${CERES_LIBRARIES}")

include_directories(${CERES_INCLUDE_DIRS})
link_directories(${CERES_LIBRARIES_DIRS})

# # 查找 OPENCV 包
set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv4")
find_package(OpenCV CONFIG REQUIRED)
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")


# 添加源文件
set(SOURCES ./src/main.cpp ./src/P2PICP.cpp)

# 添加可执行文件
add_executable(ICP ${SOURCES})

add_executable(amcl ./src/AMCL.cpp)

# 链接 PCL 库
target_link_libraries(ICP ${PCL_LIBRARIES} ${CERES_LIBRARIES})

# 可选：设置运行时路径以加载动态库
set(CMAKE_INSTALL_RPATH ${PCL_LIBRARY_DIRS})

target_link_libraries(amcl ${OpenCV_LIBRARIES})