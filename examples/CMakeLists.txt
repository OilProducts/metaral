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
