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
        %0 = ub.poison : tensor<3xi16>
        // Long form (partially poisoned)
        %1 = ub.poison #ub.poisoned_elements<arith(dense<1>)[dense<[false, true, false]>]> : tensor<3xi64>
        ```
    }];

    let arguments = (ins AnyAttr:$value);

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
            setNameFn(getResult(), "psn");
        }
    }];
}

//===----------------------------------------------------------------------===//
// FreezeOp
//===----------------------------------------------------------------------===//

def UB_FreezeOp
        : UB_Op<
            "freeze",
            [
                Pure,
                SameOperandsAndResultType,
                OpAsmOpInterface]
                # ElementwiseMappable.traits> {
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
        %psn = ub.poison : tensor<3xi16>
        %0 = ub.freeze %psn : tensor<3xi16>
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
            setNameFn(getResult(), "fzn");
        }
    }];
}

#endif
