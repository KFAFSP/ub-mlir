// RUN: ub-opt --canonicalize %s | FileCheck %s

#poison_full = #ubx.poison : bf16
#poison_defined = #ubx.poisoned_elements<arith(dense<1>)[dense<false>]> : tensor<3xi64>
#poison_partial = #ubx.poisoned_elements<arith(dense<1>)[dense<[false, true, false]>]> : tensor<3xi64>

// CHECK-LABEL: func.func @poison(
// CHECK-SAME: a0 = #ubx.poison : bf16
// CHECK-SAME: a1 = #ubx.poisoned_elements<arith(dense<1>)[dense<false>]> : tensor<3xi64>
// CHECK-SAME: a2 = #ubx.poisoned_elements<arith(dense<1>)[dense<[false, true, false]>]> : tensor<3xi64>
func.func @poison() attributes { a0 = #poison_full, a1 = #poison_defined, a2 = #poison_partial } {
    return
}
