add_library(metaral_render
    camera.cpp
    raymarch.cpp
)

set_target_properties(metaral_render PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(metaral_render
    PUBLIC
        metaral_core
        metaral_world
        metaral_sim
)

target_include_directories(metaral_render PUBLIC ${PROJECT_SOURCE_DIR}/include)
