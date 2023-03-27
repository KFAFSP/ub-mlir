//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the UB dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef UB_OPS
#define UB_OPS

include "ub-mlir/Dialect/UB/IR/Types.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/// Template for constant operations.
class UB_ConstantOpBase<string mnemonic, list<Trait> traits = []>
        : UB_Op<mnemonic, traits # [ConstantLike, Pure]>
{
    let results = (outs AnyType:$result);

    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// PoisonOp
//===----------------------------------------------------------------------===//

def UB_PoisonOp
        : UB_ConstantOpBase<
            "poison",
            [
                OpAsmOpInterface,
                DeclareOpInterfaceMethods<
                    InferTypeOpInterface,
                    ["inferReturnTypes"]>]> {
    let arguments = (ins UB_PoisonAttr:$value);

    // NOTE: The assemblyFormat does not want to work correctly...
    let hasCustomAssemblyFormat = 1;

    let hasCanonicalizer = 1;

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            setNameFn(getResult(), "poison");
        }
    }];
}

//===----------------------------------------------------------------------===//
// FreezeOp
//===----------------------------------------------------------------------===//

def UB_FreezeOp
        : UB_Op<
            "freeze",
            [Pure, SameOperandsAndResultType, OpAsmOpInterface]> {
    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$operand attr-dict `:` type($operand)";

    let hasFolder = 1;

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            setNameFn(getResult(), "frozen");
        }
    }];
}

#endif
