add_mlir_conversion_library(UBToLLVM
        UBToLLVM.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRLLVMDialect
        UBDialect
)
