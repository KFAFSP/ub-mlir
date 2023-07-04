// RUN: ub-opt --canonicalize %s | FileCheck %s

#poison_full = #ub.poison : bf16
#poison_defined = #ub.poisoned_elements<arith(dense<1>)[dense<false>]> : tensor<3xi64>
#poison_partial = #ub.poisoned_elements<arith(dense<1>)[dense<[false, true, false]>]> : tensor<3xi64>

// CHECK-LABEL: func.func @poison(
// CHECK-SAME: a0 = #ub.poison : bf16
// CHECK-SAME: a1 = #ub.poisoned_elements<arith(dense<1>)[dense<false>]> : tensor<3xi64>
// CHECK-SAME: a2 = #ub.poisoned_elements<arith(dense<1>)[dense<[false, true, false]>]> : tensor<3xi64>
func.func @poison() attributes { a0 = #poison_full, a1 = #poison_defined, a2 = #poison_partial } {
    return
}
