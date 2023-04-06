################################################################################
# ub-lsp-server
#
# The ub-lsp-server MLIR language server.
################################################################################

project(ub-lsp-server)

add_executable(${PROJECT_NAME}
    ub-lsp-server.cpp
)

if(MLIR_INCLUDE_TESTS)
    set(test_libs "MLIRTestDialect")
endif()

# Link all standard MLIR dialect and conversion libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIRLspServerLib
        ${dialect_libs}
        ${conversion_libs}
        ${test_libs}
)
