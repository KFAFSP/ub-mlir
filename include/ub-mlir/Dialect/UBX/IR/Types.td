//===- Types.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the UBX dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef UBX_TYPES
#define UBX_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "ub-mlir/Dialect/UBX/IR/Attributes.td"

//===----------------------------------------------------------------------===//
// NeverType
//===----------------------------------------------------------------------===//

def UBX_NeverType : UBX_Type<"Never"> {
    let mnemonic = "never";
    let summary = "Never type";
    let description = [{
        The `never` type is the result type of a function that never returns.

        Note that this is different from the `none` type, which is a unit type.
        The `never` type has no values at all.

        Syntax:

        ```
        never-type ::= `never`
        ```

        Example:

        ```
        !ubx.never
        ```
    }];

    let parameters = (ins);

    let assemblyFormat = "";
}

#endif
