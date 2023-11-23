add_mlir_library(UBMLIRUBXTransforms
        Freeze.cpp

    ENABLE_AGGREGATION
    DEPENDS
        UBMLIRUBXPassesIncGen
    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        UBMLIRUBXInterfaces
)
