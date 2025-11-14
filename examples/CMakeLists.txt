add_executable(coord_sanity
    coord_sanity.cpp
)

target_link_libraries(coord_sanity
    PRIVATE
        metaral_core
)

add_executable(world_fill
    world_fill.cpp
)

target_link_libraries(world_fill
    PRIVATE
        metaral_world
)

add_executable(metaral_minimal_viewer
    minimal_viewer.cpp
)

target_link_libraries(metaral_minimal_viewer
    PRIVATE
        metaral_platform
        metaral_world
        metaral_render
        metaral_core
        SDL3::SDL3
)
