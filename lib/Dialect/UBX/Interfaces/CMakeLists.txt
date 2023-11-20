add_mlir_library(UBXInterfaces
        FreezableTypeInterface.cpp
        PoisonAttrInterface.cpp

    ENABLE_AGGREGATION
    DEPENDS
        UBXIncGen
)
