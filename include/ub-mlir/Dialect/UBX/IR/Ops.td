//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the UBX dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef UBX_OPS
#define UBX_OPS

include "ub-mlir/Dialect/UBX/IR/Types.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/// Template for constant operations.
class UBX_ConstantOpBase<string mnemonic, list<Trait> traits = []>
        : UBX_Op<mnemonic, traits # [ConstantLike, Pure]>
{
    let results = (outs AnyType:$result);

    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// PoisonOp
//===----------------------------------------------------------------------===//

def UBX_PoisonOp : UBX_ConstantOpBase<
        "poison",
        [
            OpAsmOpInterface,
            DeclareOpInterfaceMethods<
                InferTypeOpInterface,
                ["inferReturnTypes"]>]> {
    let summary = "Poisoned constant operation";
    let description = [{
        The `poison` operation materializes a compile-time constant value that
        may be fully or partially poisoned, i.e., indicate undefined behavior.
        If the represented value is well-defined, it will be canonicalized to
        the source attribute materialization.

        Syntax:

        ```
        poison-op ::= `poison` $value
        ```

        Examples:

        ```
        // Short form
        %0 = ubx.poison : tensor<3xi16>
        // Long form (partially poisoned)
        %1 = ubx.poison #ub.poisoned_elements<arith(dense<1>)[dense<[false, true, false]>]> : tensor<3xi64>
        ```
    }];

    let arguments = (ins UBX_PoisonAttrInterface:$value);

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

def UBX_FreezeOp : UBX_Op<
        "freeze",
        [
            Pure,
            SameOperandsAndResultType,
            OpAsmOpInterface] # ElementwiseMappable.traits> {
    let summary = "Concretization of a poisoned value";
    let description = [{
        The `freeze` operation concretizes a potentially poisoned value. If the
        input operand is well-defined, the freeze operation is folded away.
        Otherwise, it remains, and instanciates an arbitrary but fixed value of
        its result type. Thus, it stops the propagation of poisoned values.

        Syntax:

        ```
        freeze-op ::= `freeze` $operand `:` type
        ```

        Examples:

        ```
        %poison = ubx.poison : tensor<3xi16>
        %0 = ubx.freeze %poison : tensor<3xi16>
        ```
    }];

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
