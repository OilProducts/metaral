add_subdirectory(core)
add_subdirectory(world)
add_subdirectory(sim)
add_subdirectory(render)
add_subdirectory(platform)
add_subdirectory(io)
add_subdirectory(tools)

add_library(metaral_engine INTERFACE)

target_link_libraries(metaral_engine
    INTERFACE
        metaral_core
        metaral_world
        metaral_sim
        metaral_render
        metaral_platform
        metaral_io
        metaral_tools
)

target_include_directories(metaral_engine
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)
