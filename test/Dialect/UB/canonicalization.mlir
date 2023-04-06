// RUN: ub-opt --canonicalize %s | FileCheck %s

//===----------------------------------------------------------------------===//
// poison
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @poison_unpoison(
func.func @poison_unpoison() -> i64 {
    // CHECK-DAG: %[[CST1:.+]] = arith.constant 1 : i64
    %0 = ub.poison #ub.poison<arith(1)> : i64
    // CHECK: return %[[CST1]]
    return %0 : i64
}

//===----------------------------------------------------------------------===//
// freeze
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @freeze_chain(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @freeze_chain(%arg0: i64) -> i64 {
    %0 = ub.freeze %arg0 : i64
    // CHECK-DAG: %[[FREEZE:.+]] = ub.freeze %[[ARG0]]
    %1 = ub.freeze %0 : i64
    %2 = ub.freeze %1 : i64
    // CHECK: return %[[FREEZE]]
    return %2 : i64
}

// CHECK-LABEL: @freeze_well_defined(
func.func @freeze_well_defined() -> i64 {
    // CHECK-DAG: %[[CST1:.+]] = arith.constant 1 : i64
    %cst1 = arith.constant 1 : i64
    %0 = ub.freeze %cst1 : i64
    // CHECK: return %[[CST1]]
    return %0 : i64
}

// CHECK-LABEL: @freeze_poison(
func.func @freeze_poison() -> i64 {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i64
    %cst1 = ub.poison : i64
    // CHECK: %[[FROZEN:.+]] = ub.freeze %[[POISON]]
    %0 = ub.freeze %cst1 : i64
    // CHECK: return %[[FROZEN]]
    return %0 : i64
}

//===----------------------------------------------------------------------===//
// never
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @never_propagation(
func.func @never_propagation(%arg0: i64) -> i64 {
    // CHECK-DAG: %[[NEVER:.+]] = ub.never : i64
    %never0 = ub.never : i64
    %0 = arith.muli %arg0, %arg0 : i64
    %1 = arith.addi %0, %never0 : i64
    %2 = arith.muli %1, %1 : i64
    // CHECK: return %[[NEVER]] : i64 {ub.unreachable}
    return %2 : i64
}
