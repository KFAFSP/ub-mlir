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

#endif
