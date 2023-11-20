//===- FreezableTypeInterface.td ---------------------------*- tablegen -*-===//
//
// This is the definitions file for the UBX FreezableTypeInterface interface.
//
//===----------------------------------------------------------------------===//

#ifndef UBX_FREEZABLETYPEINTERFACE
#define UBX_FREEZABLETYPEINTERFACE

include "mlir/IR/OpBase.td"

def UBX_FreezableTypeInterface : TypeInterface<"FreezableTypeInterface"> {
    let cppNamespace = "::mlir::ubx";

    let description = [{
        Interface for a type that supports freezing poison values.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Materializes any well-defined value that could be produced by
                this type, if possible.
            }],
            /*retTy=*/"::mlir::Value",
            /*methodName=*/"freeze",
            /*args=*/(ins "OpBuilder &":$builder, "Location":$location, "Attribute":$hint),
            /*methodBody*/"",
            /*defaultImplementation=*/"">];
}

#endif
