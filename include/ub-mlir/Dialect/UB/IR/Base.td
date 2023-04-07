//===- Base.td ---------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the UB dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef UB_BASE
#define UB_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def UB_Dialect : Dialect {
    let name = "ub";
    let cppNamespace = "::mlir::ub";

    let summary = "The undefined behavior dialect";
    let description = [{
        A dialect that provides attributes and operations to reason about
        undefined behavior at compile-time.
    }];

    let hasConstantMaterializer = 1;
    let hasOperationAttrVerify = 1;
    let hasCanonicalizer = 1;
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
    let useFoldAPI = kEmitFoldAdaptorFolder;

    code extraClassDeclaration = [{
    private:
        void registerAttributes();
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
class UB_Attr<string name, list<Trait> traits = []>
        : AttrDef<UB_Dialect, name, traits>;
// Template for ops.
class UB_Op<string mnemonic, list<Trait> traits = []>
        : Op<UB_Dialect, mnemonic, traits>;
// Template for types.
class UB_Type<string name, list<Trait> traits = []>
        : TypeDef<UB_Dialect, name, traits>;

#endif
