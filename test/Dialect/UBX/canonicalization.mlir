// RUN: ub-opt --canonicalize %s | FileCheck %s

//===----------------------------------------------------------------------===//
// poison
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @poison_unpoison(
func.func @poison_unpoison() -> tensor<1xi64> {
    // CHECK-DAG: %[[CST1:.+]] = arith.constant dense<1> : tensor<1xi64>
    %0 = ubx.poison #ubx.poisoned_elements<arith(dense<1>)[dense<false>]> : tensor<1xi64>
    // CHECK: return %[[CST1]]
    return %0 : tensor<1xi64>
}

//===----------------------------------------------------------------------===//
// freeze
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @freeze_chain(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @freeze_chain(%arg0: i64) -> i64 {
    %0 = ubx.freeze %arg0 : i64
    // CHECK-DAG: %[[FREEZE:.+]] = ubx.freeze %[[ARG0]]
    %1 = ubx.freeze %0 : i64
    %2 = ubx.freeze %1 : i64
    // CHECK: return %[[FREEZE]]
    return %2 : i64
}

// CHECK-LABEL: @freeze_well_defined(
func.func @freeze_well_defined() -> i64 {
    // CHECK-DAG: %[[CST1:.+]] = arith.constant 1 : i64
    %cst1 = arith.constant 1 : i64
    %0 = ubx.freeze %cst1 : i64
    // CHECK: return %[[CST1]]
    return %0 : i64
}

// CHECK-LABEL: @freeze_poison(
func.func @freeze_poison() -> i64 {
    // CHECK-DAG: %[[POISON:.+]] = ubx.poison : i64
    %cst1 = ubx.poison : i64
    // CHECK: %[[FROZEN:.+]] = ubx.freeze %[[POISON]]
    %0 = ubx.freeze %cst1 : i64
    // CHECK: return %[[FROZEN]]
    return %0 : i64
}

//===----------------------------------------------------------------------===//
// never
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @never(
func.func @never() -> !ubx.never {
    // CHECK: %[[NEVER:.+]] = ubx.never
    %never = ubx.never : !ubx.never
    // CHECK: return %[[NEVER]]
    return %never : !ubx.never
}
