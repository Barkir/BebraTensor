cmake_minimum_required(VERSION 3.14)
project(BebraTensor VERSION 0.1.0 LANGUAGES CXX C)

# ============================================================================
# C++ Standard & Compiler Settings
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Options
# ============================================================================
option(BEBRA_BUILD_TESTS "Build test suite" ON)
option(BEBRA_BUILD_EXAMPLES "Build example applications" ON)
option(BEBRA_ENABLE_SANITIZERS "Enable AddressSanitizer/UBSan for debug builds" OFF)

# ============================================================================
# Find Dependencies
# ============================================================================
find_package(Protobuf REQUIRED)
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)


include(FetchContent)
if(BEBRA_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# ============================================================================


set(BEBRA_BASE_FLAGS
    -Wall -Wextra
    -Wc++14-compat
    -Wmissing-declarations
    -Wcast-align
    -Wcast-qual
    -Wchar-subscripts
    -Wconversion
    -Wctor-dtor-privacy
    -Wempty-body
    -Wfloat-equal
    -Wformat-security
    -Winline
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wpacked
    -Wpointer-arith
    -Winit-self
    -Wredundant-decls
    -Wshadow
    -Wsign-conversion
    -Wsign-promo
    -Wstrict-overflow=2
    -Wsuggest-override
    -Wswitch-default
    -Wundef
    -Wunreachable-code
    -Wunused
    -Wvariadic-macros
    -Wno-missing-field-initializers
    -Wno-narrowing
    -Wno-old-style-cast
    -Wno-varargs
    -Wstack-protector
    -fcheck-new
    -fsized-deallocation
    -fstack-protector
    -fstrict-overflow
    -fno-omit-frame-pointer
    -fPIE
    -Werror=vla
)


if(BEBRA_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        set(SANITIZER_FLAGS "-fsanitize=address,undefined")
        list(APPEND BEBRA_BASE_FLAGS ${SANITIZER_FLAGS})
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZER_FLAGS}")
    endif()
endif()

# ============================================================================
# Core Library: Bebra
# ============================================================================
add_library(bebra_core STATIC)

target_compile_definitions(bebra_core PUBLIC
    PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

set(BEBRA_SRC_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/passes
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mlir
)
set(BEBRA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bebra/core
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bebra/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bebra/passes
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bebra/mlir
)

set(BEBRA_SOURCES "")
foreach(dir ${BEBRA_SRC_DIRS})
    file(GLOB_RECURSE dir_sources "${dir}/*.cpp")
    list(APPEND BEBRA_SOURCES ${dir_sources})
endforeach()

set(BEBRA_INCLUDES "")
foreach(dir ${BEBRA_INCLUDE_DIRS})
    file(GLOB_RECURSE dir_sources "${dir}/*.hpp")
    list(APPEND BEBRA_INCLUDES ${dir_sources})
endforeach()

set(PROTO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_proto/onnx.proto3.pb.cc
)

target_sources(bebra_core
    PRIVATE
        ${BEBRA_SOURCES}
        ${PROTO_SOURCES}
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            ${BEBRA_INCLUDES}
            ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_proto/onnx.proto3.pb.h
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
target_link_libraries(bebra_core PRIVATE
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRFuncDialect
)

target_compile_options(bebra_core PRIVATE
    ${BEBRA_BASE_FLAGS}
    -Werror
    -Wno-error=unused-parameter
)

set_source_files_properties(
    ${PROTO_SOURCES}
    PROPERTIES
        COMPILE_FLAGS "-w -Wno-error"
)


# ============================================================================
# Main Executable
# ============================================================================
add_executable(bebra_tensor src/main.cpp)

target_link_libraries(bebra_tensor PRIVATE bebra_core)

# ← ВАЖНО: Исполняемый файл тоже должен иметь флаги санитайзеров для линковки!
if(BEBRA_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(bebra_tensor PRIVATE ${SANITIZER_FLAGS})
        # target_link_options не нужен, т.к. мы задали CMAKE_EXE_LINKER_FLAGS глобально
    endif()
endif()

# ============================================================================
# Testing System
# ============================================================================
if(BEBRA_BUILD_TESTS)
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

        # Тесты тоже должны быть с санитайзерами, если они включены
        if(BEBRA_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
            if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
                target_compile_options(bebra_tests PRIVATE ${SANITIZER_FLAGS})
            endif()
        endif()

        include(GoogleTest)
        gtest_discover_tests(bebra_tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )

        add_custom_target(test_verbose
            COMMAND ${CMAKE_CTEST_COMMAND} --verbose
            DEPENDS bebra_tests
        )
    else()
        message(WARNING "No test files found in tests/ directory")
    endif()

    set(BEBRA_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
    if(EXISTS ${BEBRA_TEST_DATA_DIR})
        target_compile_definitions(bebra_tests PRIVATE
            BEBRA_TEST_DATA_DIR="${BEBRA_TEST_DATA_DIR}"
        )
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
if(BEBRA_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "  Sanitizer Flags: ${SANITIZER_FLAGS}")
endif()
message(STATUS "")
