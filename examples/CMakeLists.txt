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

if(METARAL_ENABLE_VULKAN)
    add_executable(metaral_vulkan_viewer
        vulkan_viewer.cpp
    )

    target_link_libraries(metaral_vulkan_viewer
        PRIVATE
            metaral_platform
            metaral_world
            metaral_render
            metaral_core
            SDL3::SDL3
    )
endif()
