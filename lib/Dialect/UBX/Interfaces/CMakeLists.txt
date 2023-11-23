add_mlir_library(UBMLIRUBXInterfaces
        FreezableTypeInterface.cpp
        PoisonAttrInterface.cpp

    ENABLE_AGGREGATION
    DEPENDS
        UBMLIRUBXIncGen
)
