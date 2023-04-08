//===- ControlFlowTerminatorOpInterface.td -----------------*- tablegen -*-===//
//
// This is the definitions file for the UB ControlFlowTerminatorOpInterface
// interface.
//
//===----------------------------------------------------------------------===//

#ifndef UB_CONTROLFLOWTERMINATOROPINTERFACE
#define UB_CONTROLFLOWTERMINATOROPINTERFACE

include "mlir/IR/OpBase.td"

def UB_ControlFlowTerminatorOpInterface
        : OpInterface<"ControlFlowTerminatorOpInterface"> {
    let cppNamespace = "::mlir::ub";

    let description = [{
        Interface for an operation that is a terminator of a block or region
        with well-defined control flow.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Determines whether this terminator is known to be unreachable.
            }],
            /*retTy=*/"bool",
            /*methodName=*/"isKnownUnreachable",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return control_flow_terminator_op_interface_defaults::isKnownUnreachable(*this);
            }]>,
        InterfaceMethod<
            /*desc=*/"Marks this terminator as known to be unreachable.",
            /*retTy=*/"bool",
            /*methodName=*/"markAsUnreachable",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return control_flow_terminator_op_interface_defaults::markAsUnreachable(*this);
            }]>];

    code verify = [{
        static_assert(
            ConcreteOp::template hasTrait<::mlir::OpTrait::IsTerminator>(),
            "`ControlFlowTerminatorOpInterface` is only applicable to `IsTerminator` ops."
        );

        return ::mlir::success();
    }];
}

#endif
