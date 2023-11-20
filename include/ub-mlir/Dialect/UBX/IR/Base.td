//===- Base.td ---------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the UBX dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef UBX_BASE
#define UBX_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def UBX_Dialect : Dialect {
    let name = "ubx";
    let cppNamespace = "::mlir::ubx";

    let summary = "The undefined behavior dialect";
    let description = [{
        A dialect that provides attributes and operations to reason about
        undefined behavior at compile-time.
    }];

    let hasConstantMaterializer = 1;
    let useDefaultAttributePrinterParser = 1;

    code extraClassDeclaration = [{
    private:
        void registerAttributes();
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
class UBX_Attr<string name, list<Trait> traits = []>
        : AttrDef<UBX_Dialect, name, traits>;
// Template for ops.
class UBX_Op<string mnemonic, list<Trait> traits = []>
        : Op<UBX_Dialect, mnemonic, traits>;
// Template for types.
class UBX_Type<string name, list<Trait> traits = []>
        : TypeDef<UBX_Dialect, name, traits>;

#endif
