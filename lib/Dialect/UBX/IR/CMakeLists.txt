add_mlir_dialect_library(UBXDialect
        Attributes.cpp
        Base.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        UBXIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
)
