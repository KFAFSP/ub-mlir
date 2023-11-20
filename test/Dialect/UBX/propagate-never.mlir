// RUN: ub-opt %s --ubx-propagate-never="through-ops=always" | FileCheck %s --check-prefix=OP-ALWAYS
// RUN: ub-opt %s --ubx-propagate-never="through-ops=pure" | FileCheck %s --check-prefix=OP-PURE
// RUN: ub-opt %s --ubx-propagate-never="through-ops=constexpr" | FileCheck %s --check-prefix=OP-CONSTEXPR

// OP-ALWAYS-LABEL: func.func @through_impure_op(
// OP-ALWAYS-SAME: %[[ARG0:.+]]: memref
// OP-PURE-LABEL: func.func @through_impure_op(
// OP-PURE-SAME: %[[ARG0:.+]]: memref
// OP-CONSTEXPR-LABEL: func.func @through_impure_op(
// OP-CONSTEXPR-SAME: %[[ARG0:.+]]: memref
func.func @through_impure_op(%arg0: memref<1xi32>) -> i32 {
    // OP-ALWAYS: %[[NEVER:.+]] = ubx.never : i32
    // OP-PURE: %[[NEVER:.+]] = ubx.never : index
    // OP-CONSTEXPR: %[[NEVER:.+]] = ubx.never : index
    %never = ubx.never : index
    // OP-PURE: %[[LOAD:.+]] = memref.load %[[ARG0]][%[[NEVER]]]
    // OP-CONSTEXPR: %[[LOAD:.+]] = memref.load %[[ARG0]][%[[NEVER]]]
    %0 = memref.load %arg0[%never] : memref<1xi32>
    // OP-ALWAYS: return %[[NEVER]]
    // OP-PURE: return %[[LOAD]]
    // OP-CONSTEXPR: return %[[LOAD]]
    return %0 : i32
}

// OP-ALWAYS-LABEL: func.func @through_pure_op(
// OP-ALWAYS-SAME: %[[ARG0:.+]]: i32
// OP-PURE-LABEL: func.func @through_pure_op(
// OP-PURE-SAME: %[[ARG0:.+]]: i32
// OP-CONSTEXPR-LABEL: func.func @through_pure_op(
// OP-CONSTEXPR-SAME: %[[ARG0:.+]]: i32
func.func @through_pure_op(%arg0: i32) -> i32 {
    // OP-ALWAYS: %[[NEVER:.+]] = ubx.never : i32
    // OP-PURE: %[[NEVER:.+]] = ubx.never : i32
    // OP-CONSTEXPR: %[[NEVER:.+]] = ubx.never : i32
    %never = ubx.never : i32
    // OP-CONSTEXPR: %[[ADD:.+]] = arith.addi %[[ARG0]], %[[NEVER]]
    %0 = arith.addi %arg0, %never : i32
    // OP-ALWAYS: return %[[NEVER]]
    // OP-PURE: return %[[NEVER]]
    // OP-CONSTEXPR: return %[[ADD]]
    return %0 : i32
}

// OP-ALWAYS-LABEL: func.func @through_constexpr_op(
// OP-PURE-LABEL: func.func @through_constexpr_op(
// OP-CONSTEXPR-LABEL: func.func @through_constexpr_op(
func.func @through_constexpr_op() -> i32 {
    // OP-ALWAYS: %[[NEVER:.+]] = ubx.never : i32
    // OP-PURE: %[[NEVER:.+]] = ubx.never : i32
    // OP-CONSTEXPR: %[[NEVER:.+]] = ubx.never : i32
    %never = ubx.never : i32
    %cst1 = arith.constant 1 : i32
    %0 = arith.addi %never, %cst1 : i32
    // OP-ALWAYS: return %[[NEVER]]
    // OP-PURE: return %[[NEVER]]
    // OP-CONSTEXPR: return %[[NEVER]]
    return %0 : i32
}
