add_mlir_library(UBXInterfaces
        PoisonAttrInterface.cpp

    ENABLE_AGGREGATION
)

add_dependencies(UBXInterfaces UBXIncGen)
