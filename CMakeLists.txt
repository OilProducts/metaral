cmake_minimum_required(VERSION 3.23)

project(MetaralEngine VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(METARAL_PLATFORM_BACKEND "SDL3" CACHE STRING "Platform backend implementation")
set_property(CACHE METARAL_PLATFORM_BACKEND PROPERTY STRINGS SDL3)

option(METARAL_BUNDLE_SDL3 "Download/build SDL3 locally if a system package is not found" ON)

if(METARAL_PLATFORM_BACKEND STREQUAL "SDL3")
    find_package(SDL3 QUIET CONFIG)

    if(NOT SDL3_FOUND)
        if(NOT METARAL_BUNDLE_SDL3)
            message(FATAL_ERROR "SDL3 not found. Set SDL3_DIR or enable METARAL_BUNDLE_SDL3 to fetch it automatically.")
        endif()

        include(FetchContent)

        set(SDL_SHARED OFF CACHE BOOL "Build SDL3 as shared library" FORCE)
        set(SDL_STATIC ON CACHE BOOL "Build SDL3 static library" FORCE)
        set(SDL_TEST_LIBRARY OFF CACHE BOOL "Disable SDL3 test library" FORCE)
        set(SDL_DISABLE_INSTALL ON CACHE BOOL "Disable SDL3 install" FORCE)

        FetchContent_Declare(
            SDL3
            GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
            GIT_TAG release-3.2.26
            GIT_SHALLOW TRUE
        )

        FetchContent_MakeAvailable(SDL3)
    endif()
else()
    message(FATAL_ERROR "Unsupported METARAL_PLATFORM_BACKEND: ${METARAL_PLATFORM_BACKEND}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)

option(METARAL_BUILD_EXAMPLES "Build sample applications in examples/" ON)
if(METARAL_BUILD_EXAMPLES AND EXISTS "${PROJECT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()
