add_library(metaral_io INTERFACE)

target_link_libraries(metaral_io
    INTERFACE
        metaral_core
        metaral_world
        metaral_sim
)

target_include_directories(metaral_io INTERFACE ${PROJECT_SOURCE_DIR}/include)
