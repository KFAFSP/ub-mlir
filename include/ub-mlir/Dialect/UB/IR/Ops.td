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
        poison-op ::= `poison` $value `:` type
        ```

        Examples:

        ```
        // Fully poisoned value.
        %0 = ub.poison : tensor<3xi16>
        // Partially poisoned value.
        %1 = ub.poison #ub.poison<"BEEF", arith(dense<2>)> : tensor<32xi64>
        // Well-defined value (will be canonicalized to arith.constant).
        %1 = ub.poison #ub.poison<arith(dense<2>)> : tensor<32xi64>
        ```
    }];

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
        freeze-op ::= `freeze` $operand attr-dict `:` type
        ```

        Examples:

        ```
        %0 = ub.freeze %poison : tensor<3xi16>
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

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

def UB_UnreachableOp : UB_Op<"unreachable", [Pure, Terminator]> {
    let summary = "Unreachable SSACFG terminator";
    let description = [{
        The `unreachable` operation terminates an SSACFG block without transfer
        of control flow to a successor. Additionally, it requires that its code
        path can never be executed.

        Syntax:

        ```
        unreachable-op ::= `unreachable`
        ```

        Examples:

        ```
        ^bb0:
            ub.unreachable
        ```
    }];

    let arguments = (ins);
    let results = (outs);

    let assemblyFormat = "attr-dict";

    let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// NeverOp
//===----------------------------------------------------------------------===//

def UB_NeverOp : UB_ConstantOpBase<"never", [OpAsmOpInterface]> {
    let summary = "Unreachable value constant operation";
    let description = [{
        The `never` operation never produces a result, i.e., is an operation
        from which control flow never returns. It materializes an unreachable
        value of any type. If no type is specified, the result type is `never`.

        Syntax:

        ```
        never-op ::= `never` attr-dict (`:` type)?
        ```

        Examples:

        ```
        %0 = ub.never
        %1 = ub.never : i64
        ```
    }];

    let arguments = (ins);
    let results = (outs AnyType:$result);

    let builders = [
        OpBuilder<(ins CArg<"Type", "{}">:$result),
        [{
            if (!result) result = NeverType::get($_builder.getContext());
            $_state.addTypes(result);
        }]>
    ];
    let skipDefaultBuilders = 1;

    let assemblyFormat = "attr-dict custom<MaybeNever>(type($result))";

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            setNameFn(getResult(), "never");
        }
    }];
}

#endif
