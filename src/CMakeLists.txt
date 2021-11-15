add_subdirectory(util)
add_subdirectory(map_render)
add_subdirectory(tool)
add_subdirectory(optimization)
add_subdirectory(relocalization)
add_subdirectory(full_system)

set(DSL_DEPENDENCIES
dsl_full_system
dsl_relocalization
dsl_cost_functor
dsl_util
dsl_tool
dsl_map_render
${ThreeParty_LIBS}
gmock gtest gtest_main
)

set(DSL_LIBRARIES
dsl_full_system
dsl_relocalization
dsl_cost_functor
dsl_util
dsl_tool
dsl_map_render
${ThreeParty_LIBS}
${Google_LIBRARIES}
${Ext_LIBRARIES}
)

add_executable(dsl_main dsl_main.cc)
add_dependencies(dsl_main ${DSL_DEPENDENCIES})
target_link_libraries(dsl_main
${DSL_LIBRARIES}
)

add_executable(dsl_full_main dsl_full_main.cc)
add_dependencies(dsl_full_main ${DSL_DEPENDENCIES})
target_link_libraries(dsl_full_main
${DSL_LIBRARIES}
)

add_executable(colorizer colorizer.cc)
add_dependencies(colorizer ${DSL_DEPENDENCIES})
target_link_libraries(colorizer
${DSL_LIBRARIES}
)

add_executable(depth_generator depth_generator.cc)
add_dependencies(depth_generator ${DSL_DEPENDENCIES})
target_link_libraries(depth_generator
${DSL_LIBRARIES}
)

add_executable(depth_generator_from_map depth_generator_from_map.cc)
add_dependencies(depth_generator_from_map ${DSL_DEPENDENCIES})
target_link_libraries(depth_generator_from_map
${DSL_LIBRARIES}
)

add_executable(desc_map_builder desc_map_builder.cc)
add_dependencies(desc_map_builder ${DSL_DEPENDENCIES})
target_link_libraries(desc_map_builder
${DSL_LIBRARIES}
)

add_executable(desc_map_clique desc_map_clique.cc)
add_dependencies(desc_map_clique ${DSL_DEPENDENCIES})
target_link_libraries(desc_map_clique
${DSL_LIBRARIES}
)

add_executable(desc_map_keyframe desc_map_keyframe.cc)
add_dependencies(desc_map_keyframe ${DSL_DEPENDENCIES})
target_link_libraries(desc_map_keyframe
${DSL_LIBRARIES}
)