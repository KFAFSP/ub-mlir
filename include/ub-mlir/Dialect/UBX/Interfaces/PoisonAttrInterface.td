//===- PoisonAttrInterface.td ------------------------------*- tablegen -*-===//
//
// This is the definitions file for the UBX PoisonAttrInterface interface.
//
//===----------------------------------------------------------------------===//

#ifndef UBX_POISONATTRINTERFACE
#define UBX_POISONATTRINTERFACE

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

def UBX_PoisonAttrInterface
        : AttrInterface<"PoisonAttrInterface", [TypedAttrInterface]> {
    let cppNamespace = "::mlir::ubx";

    let description = [{
        Interface for an attribute that holds a potentially poisoned value.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Indicates whether this attribute holds a poisoned value.
            }],
            /*retTy=*/"bool",
            /*methodName=*/"isPoisoned",
            /*args=*/(ins ),
            /*methodBody*/"",
            /*defaultImplementation=*/[{
                return true;
            }]>,
        InterfaceMethod<
            /*desc=*/[{
                Materializes any well-defined value that could be produced by
                this attribute, if possible.

                In particular, if `isPoisoned()` returns `false`, the attribute
                is required to materialize the contained well-defined value.
            }],
            /*retTy=*/"::mlir::Value",
            /*methodName=*/"freeze",
            /*args=*/(ins "OpBuilder &":$builder, "Location":$location),
            /*methodBody*/"",
            /*defaultImplementation=*/[{
                return {};
            }]>];
}

#endif
