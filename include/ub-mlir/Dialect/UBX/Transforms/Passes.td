//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the UBX passes.
//
//===----------------------------------------------------------------------===//

#ifndef UBX_PASSES
#define UBX_PASSES

include "mlir/Pass/PassBase.td"

def Freeze : Pass<"ubx-freeze"> {
    let summary = "Instantiates well-defined values for `ubx.freeze` results";
    let description = [{
        An `ubx.freeze` operation inhibits the propagation of poison values by
        requring to have an arbitrary but well-defined result. The concrete
        value instantiated in this way is unspecified.

        If a `ubx.freeze` operation is statically-known to receive a poisoned
        operand, and it was impossible to eliminate the `ubx.freeze` operation
        following a guarantee of its producer, a value must be created.

        This pass attempts to replace the result of every `ubx.freeze` operation
        operating on a constant operand with a frozen value, by trying the two
        following interface methods in order:

            - `PoisonAttrInterface::freeze(OpBuilder &, Location)`

              Lets the poisoned attribute try to manifest a value.

            - `FreezableTypeInterface::freeze(OpBuilder &, Location, Attribute)`

              Lets the poisoned type try to manifest a value, using the constant
              operand as a hint.
    }];

    let constructor = "::mlir::ubx::createFreezePass()";
}

def PropagateNever : Pass<"ubx-propagate-never"> {
    let summary = "Propagates `#ubx.never` values across SSA dependencies";
    let description = [{
        An `#ubx.never` values indicates the result of an operation that is
        statically known to never be executed. Therefore, operations that
        depend on such values are transitively unreachable.

        When an unreachable operation becomes no-return, it propagates a
        never value to all of its results. This carries the unreachability in
        the IR, through SSA dependencies. This pass performs such propagation in
        the following cicrumstances:

        - Through operations

          According to the `through-ops` option, any operation will propagate
          never that matches the specified filter:

          - `always` affects all operations.

          - `pure` affects only pure operations (i.e., that have no memory
             effects and are speculatable).

          - `constexpr` affects only `pure` operations that have only constant
            operands but cannot be folded. This is the default.

        TODO: Propagate based on NeverType
        TODO: Propagate using RegionBranchOpInterface
        TODO: Propagate through CallOpInterface
        TODO: Propagate to private FunctionOpInterface return type
    }];

    let options = [
        Option<
            "throughOps",
            "through-ops",
            "::mlir::ubx::OpNeverPropagation",
	        /*default=*/"::mlir::ubx::OpNeverPropagation::Constexpr",
	        "Propagate never values through operations",
	        [{::llvm::cl::values(
                clEnumValN(::mlir::ubx::OpNeverPropagation::Never, "never", "Never"),
                clEnumValN(::mlir::ubx::OpNeverPropagation::Constexpr, "constexpr", "When unfoldable constant expression (Default)"),
                clEnumValN(::mlir::ubx::OpNeverPropagation::Pure, "pure", "When pure"),
                clEnumValN(::mlir::ubx::OpNeverPropagation::Always, "always", "Always")
            )}]>];

    let constructor = "::mlir::ubx::createPropagateNeverPass()";
}

#endif
