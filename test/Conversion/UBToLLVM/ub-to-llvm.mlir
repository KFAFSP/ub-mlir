// RUN: ub-opt --convert-ub-to-llvm %s | FileCheck %s

// CHECK-LABEL: func.func @poison(
func.func @poison() -> i64 {
    // CHECK: %[[POISON:.+]] = llvm.mlir.poison : i64
    %poison = ub.poison : i64
    // CHECK: return %[[POISON]]
    return %poison : i64
}

// CHECK-LABEL: func.func @freeze(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @freeze(%arg0: i64) -> i64 {
    // CHECK: %[[FREEZE:.+]] = llvm.freeze %[[ARG0]]
    %0 = ub.freeze %arg0 : i64
    // CHECK: return %[[FREEZE]]
    return %0 : i64
}

// CHECK-LABEL: func.func @unreachable(
func.func @unreachable(%arg0: i64) -> i64 {
    // CHECK: cf.switch
    cf.switch %arg0 : i64, [
        // CHECK-NEXT: default: ^[[BB0:.+]],
        default: ^bb0,
        1: ^bb1
    ]

// CHECK: ^[[BB0]]:
^bb0:
    // CHECK: llvm.unreachable
    ub.unreachable

^bb1:
    return %arg0 : i64
}
