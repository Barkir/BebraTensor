cmake_minimum_required(VERSION 3.14)
project(BebraTensor VERSION 0.1.0 LANGUAGES CXX)

# ============================================================================
# C++ Standard & Compiler Settings
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Options
# ============================================================================
option(BEBRA_BUILD_TESTS "Build test suite" ON)
option(BEBRA_BUILD_EXAMPLES "Build example applications" ON)
option(BEBRA_ENABLE_SANITIZERS "Enable AddressSanitizer for debug builds" OFF)

# ============================================================================
# Find Dependencies
# ============================================================================
# Protobuf for ONNX support
find_package(Protobuf REQUIRED)

# GoogleTest for testing (fetch if not found)
include(FetchContent)
if(BEBRA_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# ============================================================================
# Core Library: Bebra
# ============================================================================
add_library(bebra_core STATIC)

target_compile_definitions(bebra_core PUBLIC
    PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)


target_sources(bebra_core
    PRIVATE
        src/BebraGraph.cpp
        src/BebraAttr.cpp
        src/BebraColors.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_proto/onnx.proto3.pb.cc
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/bebra/core/BebraAttr.hpp
            include/bebra/core/BebraColors.hpp
            include/bebra/core/BebraErr.hpp
            include/bebra/core/BebraGraph.hpp
            include/bebra/core/BebraNode.hpp
            include/bebra/core/BebraTensor.hpp
            include/bebra/core/BebraConst.hpp
            include/bebra/ops/BebraOperators.hpp
            include/onnx_proto/onnx.proto3.pb.h
)

target_include_directories(bebra_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(bebra_core
    PUBLIC
        protobuf::libprotobuf
)

# Compiler-specific optimizations
if(MSVC)
    target_compile_options(bebra_core PRIVATE /W4 /WX-)
else()
    target_compile_options(bebra_core PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -O2
        -march=native
    )
endif()

# ============================================================================
# Main Executable
# =================l===========================================================
add_executable(bebra_tensor src/main.cpp)
target_link_libraries(bebra_tensor PRIVATE bebra_core)

# ============================================================================
# Testing System
# ============================================================================
if(BEBRA_BUILD_TESTS)
    # Collect all test files
    file(GLOB TEST_SOURCES tests/*.cpp tests/**/*.cpp)

    if(TEST_SOURCES)
        add_executable(bebra_tests ${TEST_SOURCES})

        target_link_libraries(bebra_tests
            PRIVATE
                bebra_core
                GTest::gtest_main
                GTest::gmock
        )

        target_compile_definitions(bebra_tests PRIVATE
        TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        )

        # Register tests with CTest
        include(GoogleTest)
        gtest_discover_tests(bebra_tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )

        # Add custom test target for convenience
        add_custom_target(test_verbose
            COMMAND ${CMAKE_CTEST_COMMAND} --verbose
            DEPENDS bebra_tests
        )
    else()
        message(WARNING "No test files found in tests/ directory")
    endif()

    # Optional: Add test data directory
    set(BEBRA_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
    if(EXISTS ${BEBRA_TEST_DATA_DIR})
        target_compile_definitions(bebra_tests PRIVATE
            BEBRA_TEST_DATA_DIR="${BEBRA_TEST_DATA_DIR}"
        )
    endif()
endif()

# ============================================================================
# Sanitizers (Debug builds)
# ============================================================================
if(BEBRA_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(bebra_core PRIVATE -fsanitize=address,undefined)
        target_link_options(bebra_core PRIVATE -fsanitize=address,undefined)
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(TARGETS bebra_core bebra_tensor
    EXPORT BebraTensorTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILE_SET HEADERS
)

install(EXPORT BebraTensorTargets
    FILE BebraTensorTargets.cmake
    NAMESPACE Bebra::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BebraTensor
)

# ============================================================================
# Package Configuration
# ============================================================================
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BebraTensorConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/BebraTensorConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BebraTensor
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/BebraTensorConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/BebraTensorConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/BebraTensorConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BebraTensor
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "BebraTensor Configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Tests:          ${BEBRA_BUILD_TESTS}")
message(STATUS "  Sanitizers:     ${BEBRA_ENABLE_SANITIZERS}")
message(STATUS "")
