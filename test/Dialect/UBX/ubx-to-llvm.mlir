// RUN: ub-opt --convert-ubx-to-llvm %s | FileCheck %s

// CHECK-LABEL: func.func @poison(
func.func @poison() -> i32 {
    // CHECK: %[[POISON:.+]] = llvm.mlir.poison : i32
    %poison = ubx.poison : i32
    // CHECK: return %[[POISON]]
    return %poison : i32
}

// CHECK-LABEL: func.func @freeze(
// CHECK-SAME: %[[ARG0:.+]]: i32
func.func @freeze(%arg0: i32) -> i32 {
    // CHECK: %[[FROZEN:.+]] = llvm.freeze %[[ARG0]] : i32
    %frozen = ubx.freeze %arg0 : i32
    // CHECK: return %[[FROZEN]]
    return %frozen : i32
}
