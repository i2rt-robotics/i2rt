cmake_minimum_required(VERSION 3.15)
project(i2rt_ecat LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "i2rt-ecat drives an EtherCAT NIC through Linux raw sockets and builds on Linux only")
endif()

# Upstream SOEM, compiled as shipped (its own CMake adds -Werror, which newer compilers trip).
set(SOEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/soem)
file(GLOB SOEM_SOURCES ${SOEM}/soem/*.c ${SOEM}/osal/linux/*.c ${SOEM}/oshw/linux/*.c)
add_library(soem STATIC ${SOEM_SOURCES})
set_target_properties(soem PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 99)
target_include_directories(soem PUBLIC ${SOEM}/soem ${SOEM}/osal ${SOEM}/osal/linux ${SOEM}/oshw/linux)
target_compile_options(soem PRIVATE -w)

find_package(Threads REQUIRED)
add_library(i2rt_ecat SHARED src/ecat_can_bus.cpp)
set_target_properties(i2rt_ecat PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_options(i2rt_ecat PRIVATE -Wall -Wextra)
target_link_libraries(i2rt_ecat PRIVATE soem Threads::Threads rt)
# Keep SOEM's globals out of the dynamic symbol table, so they cannot clash with anything else the
# Python process has loaded.
target_link_options(i2rt_ecat PRIVATE -Wl,--exclude-libs,ALL)
install(TARGETS i2rt_ecat LIBRARY DESTINATION i2rt_ecat)
