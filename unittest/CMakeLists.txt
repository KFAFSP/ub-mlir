################################################################################
# ub-mlir-tests
#
# The ub-mlir unit test project.
################################################################################

project(ub-mlir-tests)

find_package(doctest REQUIRED)
include(doctest)

include(CTest)

add_executable(${PROJECT_NAME}
    main.cpp
    StaticFolder.cpp
)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIRTensorDialect
        UBDialect
)
target_compile_options(${PROJECT_NAME}
    PRIVATE
        -fno-rtti
)

doctest_discover_tests(${PROJECT_NAME})
