add_library(metaral_sim INTERFACE)

target_link_libraries(metaral_sim
    INTERFACE
        metaral_core
        metaral_world
)

target_include_directories(metaral_sim INTERFACE ${PROJECT_SOURCE_DIR}/include)
