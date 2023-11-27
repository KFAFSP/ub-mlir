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
    Constant.cpp
    FoldingContext.cpp
    main.cpp
)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        UBMLIRUBXDialect
        UBMLIRExtensions
        MLIRArithDialect
)
target_compile_options(${PROJECT_NAME}
    PRIVATE
        -fno-rtti
)

doctest_discover_tests(${PROJECT_NAME})
