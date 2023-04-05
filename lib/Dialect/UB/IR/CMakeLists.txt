add_mlir_dialect_library(UBDialect
        Attributes.cpp
        Base.cpp
        Ops.cpp
        Types.cpp
        UB.cpp

    DEPENDS
        UBIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        UBAnalysis
)
