add_mlir_library(UBInterfaces
        ControlFlowTerminatorOpInterface.cpp

    ENABLE_AGGREGATION
    DEPENDS
        UBIncGen
)
