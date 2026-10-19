include(GoogleTest)

add_executable(utilsTests tests.cpp)

usd_plugin_compile_config(utilsTests)

target_link_libraries(utilsTests
                      PRIVATE
                        sdf
                        usd
                        usdGeom
                        usdSkel
                        usdShade
                        GTest::gtest
                        GTest::gtest_main
                        fileformatUtilsTest
)

add_test(NAME utilsTests
         COMMAND utilsTests
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Direct image helper regressions, with per-run temporary artifacts.
add_executable(imageIoTests imageIo.cpp)
usd_plugin_compile_config(imageIoTests)
target_link_libraries(imageIoTests PRIVATE fileformatUtils)
add_test(NAME imageIoTests
    COMMAND ${CMAKE_COMMAND} "-DTEST_EXECUTABLE=$<TARGET_FILE:imageIoTests>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/runImageIo.cmake")
