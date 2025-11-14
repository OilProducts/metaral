add_library(metaral_tools INTERFACE)

target_link_libraries(metaral_tools
    INTERFACE
        metaral_core
        metaral_world
        metaral_sim
        metaral_render
)

target_include_directories(metaral_tools INTERFACE ${PROJECT_SOURCE_DIR}/include)
