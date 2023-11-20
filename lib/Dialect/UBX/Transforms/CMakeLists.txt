add_mlir_library(UBXTransforms
        Freeze.cpp
        PropagateNever.cpp

    ENABLE_AGGREGATION
    DEPENDS
        UBXPassesIncGen
    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        UBXInterfaces
)
