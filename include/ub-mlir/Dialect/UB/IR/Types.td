//===- Types.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the UB dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef UB_TYPES
#define UB_TYPES

include "ub-mlir/Dialect/UB/IR/Attributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// NeverType
//===----------------------------------------------------------------------===//

def UB_NeverType : UB_Type<"Never"> {
    let mnemonic = "never";
    let summary = "Never type";
    let description = [{
        The `never` type is the result type of a function that never returns.

        Syntax:

        ```
        never-type ::= `never`
        ```

        Example:

        ```
        !ub.never
        ```
    }];

    let parameters = (ins);

    let assemblyFormat = "";
}

#endif
