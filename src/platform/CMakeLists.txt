add_library(metaral_platform
    sdl_platform.cpp
)

set_target_properties(metaral_platform PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(metaral_platform
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(metaral_platform
    PUBLIC
        metaral_core
        SDL3::SDL3
)
