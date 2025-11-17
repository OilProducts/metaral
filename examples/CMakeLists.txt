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
    include(FetchContent)

    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.90.8
        GIT_SHALLOW TRUE
    )

    FetchContent_GetProperties(imgui)
    if(NOT imgui_POPULATED)
        FetchContent_Populate(imgui)
    endif()

    set(METARAL_IMGUI_BACKEND_SRC "${PROJECT_SOURCE_DIR}/third_party/imgui/imgui_impl_sdl3.cpp")
    if(EXISTS "${METARAL_IMGUI_BACKEND_SRC}")
        configure_file("${METARAL_IMGUI_BACKEND_SRC}"
                       "${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp"
                       COPYONLY)
    endif()

    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    )

    target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )

    target_link_libraries(imgui PUBLIC Vulkan::Vulkan SDL3::SDL3)
    target_compile_definitions(imgui PUBLIC SDL_ENABLE_OLD_NAMES=1)

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
            imgui
    )
endif()
