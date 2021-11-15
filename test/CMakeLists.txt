set(Test_DEPENDENCIES
dsl_full_system
dsl_relocalization
dsl_cost_functor
dsl_util
dsl_tool
dsl_map_render
${ThreeParty_LIBS}
gmock gtest gtest_main
)

set(Test_LIBRARIES
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

add_executable(ref_test ref_test.cc)
target_link_libraries(ref_test
${Test_LIBRARIES}
)
add_dependencies(ref_test
${Test_DEPENDENCIES}
)

add_executable(mt_test mt_test.cc)
target_link_libraries(mt_test
${Test_LIBRARIES}
)
add_dependencies(mt_test
${Test_DEPENDENCIES}
)

add_executable(img_test img_test.cc)
target_link_libraries(img_test
${Test_LIBRARIES}
)
add_dependencies(img_test
${Test_DEPENDENCIES}
)

add_executable(coarse_tracker_test coarse_tracker_test.cc)
target_link_libraries(coarse_tracker_test
${Test_LIBRARIES}
)
add_dependencies(coarse_tracker_test
${Test_DEPENDENCIES}
)

add_executable(front_end_test front_end_test.cc)
target_link_libraries(front_end_test
${Test_LIBRARIES}
)
add_dependencies(front_end_test
${Test_DEPENDENCIES}
)

add_executable(ceres_test ceres_test.cc)
target_link_libraries(ceres_test
${Test_LIBRARIES}
)
add_dependencies(ceres_test
${Test_DEPENDENCIES}
)

add_executable(optimization_test optimization_test.cc)
target_link_libraries(optimization_test
${Test_LIBRARIES}
)
add_dependencies(optimization_test
${Test_DEPENDENCIES}
)

add_executable(decomposition_test decomposition_test.cc)
target_link_libraries(decomposition_test
${Test_LIBRARIES}
)
add_dependencies(decomposition_test
${Test_DEPENDENCIES}
)

add_executable(euroc_test euroc_test.cc)
target_link_libraries(euroc_test
${Test_LIBRARIES}
${Boost_LIBRARIES}
)
add_dependencies(euroc_test
${Test_DEPENDENCIES}
)

add_executable(render_test render_test.cc)
target_link_libraries(render_test
${Test_LIBRARIES}
${Boost_LIBRARIES}
)
add_dependencies(render_test
${Test_DEPENDENCIES}
)

add_executable(dbow2_test dbow2_test.cc)
target_link_libraries(dbow2_test
${Test_LIBRARIES}
${Boost_LIBRARIES}
DBoW2
)
add_dependencies(dbow2_test
${Test_DEPENDENCIES}
DBoW2
)

add_executable(netvlad_test netvlad_test.cc)
target_link_libraries(netvlad_test
${Test_LIBRARIES}
${Boost_LIBRARIES}
DBoW2
)
add_dependencies(netvlad_test
${Test_DEPENDENCIES}
DBoW2
)

add_executable(sp_test sp_test.cc)
target_link_libraries(sp_test
${Test_LIBRARIES}
${Boost_LIBRARIES}
)
add_dependencies(sp_test
${Test_DEPENDENCIES}
)

add_executable(desc_map_vis desc_map_vis.cc)
target_link_libraries(desc_map_vis
${Test_LIBRARIES}
${Boost_LIBRARIES}
)
add_dependencies(desc_map_vis
${Test_DEPENDENCIES}
)