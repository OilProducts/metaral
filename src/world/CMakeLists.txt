add_library(metaral_world INTERFACE)

target_link_libraries(metaral_world
    INTERFACE
        metaral_core
)

target_include_directories(metaral_world INTERFACE ${PROJECT_SOURCE_DIR}/include)
