add_library(metaral_render INTERFACE)

target_link_libraries(metaral_render
    INTERFACE
        metaral_core
        metaral_world
        metaral_sim
)

target_include_directories(metaral_render INTERFACE ${PROJECT_SOURCE_DIR}/include)
