cmake_minimum_required(VERSION 3.16)

project(cnn_ckks_project)
set(CMAKE_CXX_COMPILER g++)

option(COMP "Build comparison operation" ON)
option(BOOT "Build comparison operation" ON)
option(CNN "Build comparison operation" ON)
option(TEST "Build comparison operation" ON)
#option(COMP "Build comparison operation" OFF)
#option(BOOT "Build comparison operation" OFF)
#option(CNN "Build comparison operation" OFF)
#option(TEST "Build comparison operation" OFF)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./run)

set(CMAKE_PREFIX_PATH "$ENV{HOME}/PNP/seal-modified-3.6.6-install" ${CMAKE_PREFIX_PATH})
find_package(SEAL 3.6 REQUIRED)

# 添加子目录，并指定唯一的二进制目录
add_subdirectory(common common_build)

if(COMP)
    add_subdirectory(cpu-ckks/single-key/comp comp_build)
	add_executable(compare)
	target_sources(compare
		PRIVATE
			./run/run_compare.cpp
			${COMMON_SOURCE_FILES}
			${COMP_SOURCE_FILES}
	)
	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		message(STATUS "Building in Debug mode with -O0")
		target_compile_options(compare PRIVATE -Wall -O0 -std=c++17 -pthread -march=native)
	else(CMAKE_BUILD_TYPE STREQUAL "Release")
		message(STATUS "Building in Release mode with -O3")
		target_compile_options(compare PRIVATE -Wall -O3 -std=c++17 -pthread -march=native)
	endif()
#	target_compile_options(compare PRIVATE -Wall -O3 -std=c++17 -pthread -march=native)
	target_link_libraries(compare PRIVATE ntl gmp m pthread SEAL::seal)
	target_include_directories(compare PUBLIC /usr/local/include /usr/local/include/NTL ${COMMON_HEADER_DIR} ${COMP_HEADER_DIR})
endif()

if(BOOT)
    add_subdirectory(cpu-ckks/single-key/ckks_bootstrapping ckks_bootstrapping_build)
	add_executable(bootstrapping)
	target_sources(bootstrapping
		PRIVATE
			./run/run_bootstrapping.cpp
			${COMMON_SOURCE_FILES}
			${BOOTSTRAPPING_SOURCE_FILES}
	)

	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		message(STATUS "Building in Debug mode with -O0")
		target_compile_options(bootstrapping PRIVATE -Wall -O0 -std=c++17 -pthread -march=native)
	else(CMAKE_BUILD_TYPE STREQUAL "Release")
		message(STATUS "Building in Release mode with -O3")
		target_compile_options(bootstrapping PRIVATE -Wall -O3 -std=c++17 -pthread -march=native)
	endif()
#	target_compile_options(bootstrapping PRIVATE -Wall -O0 -std=c++17 -pthread -march=native)
	target_link_libraries(bootstrapping PRIVATE ntl gmp m pthread SEAL::seal)
	target_include_directories(bootstrapping PUBLIC /usr/local/include /usr/local/include/NTL ${COMMON_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR})
endif()

if(CNN)
    add_subdirectory(cpu-ckks/single-key/comp comp_build_cnn)
    add_subdirectory(cpu-ckks/single-key/ckks_bootstrapping ckks_bootstrapping_build_cnn)
    add_subdirectory(cpu-ckks/single-key/cnn cnn_build)
	add_executable(cnn)
	target_sources(cnn
		PRIVATE
			./run/run_cnn.cpp
			${COMMON_SOURCE_FILES}
			${CNN_SOURCE_FILES}
			${COMP_SOURCE_FILES}
			${BOOTSTRAPPING_SOURCE_FILES}
	)
	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		message(STATUS "Building in Debug mode with -O0")
		target_compile_options(cnn PRIVATE -w -O0 -std=c++17 -pthread -march=native -fopenmp)
	else(CMAKE_BUILD_TYPE STREQUAL "Release")
		message(STATUS "Building in Release mode with -O3")
		target_compile_options(cnn PRIVATE -w -O3 -std=c++17 -pthread -march=native -fopenmp)
	endif()
#	target_compile_options(cnn PRIVATE -w -O0 -std=c++17 -pthread -march=native -fopenmp)
	target_link_libraries(cnn PRIVATE ntl gmp m pthread SEAL::seal gomp)
	target_include_directories(cnn PUBLIC /usr/include /usr/local/include /usr/local/include/NTL ${COMMON_HEADER_DIR} ${CNN_HEADER_DIR} ${COMP_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR})
	# target_include_directories(cnn PUBLIC /usr/include /usr/local/include /usr/local/include/NTL ${COMMON_HEADER_DIR} ${CNN_HEADER_DIR} ${COMP_HEADER_DIR})
endif()