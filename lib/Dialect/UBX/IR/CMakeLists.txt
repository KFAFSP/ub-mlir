add_mlir_dialect_library(UBMLIRUBXDialect
        Attributes.cpp
        Base.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        UBMLIRUBXIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        UBMLIRUBXInterfaces
        UBMLIRUBXTransforms
)
