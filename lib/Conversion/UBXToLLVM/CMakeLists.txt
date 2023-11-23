add_mlir_conversion_library(UBMLIRUBXToLLVM
        UBXToLLVM.cpp

    DEPENDS
        UBMLIRConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRLLVMDialect
        UBMLIRUBXDialect
)
