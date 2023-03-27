// RUN: ub-opt --canonicalize %s | FileCheck %s

#poison_full = #ub.poison : bf16
#poison_defined = #ub.poison<arith(dense<1>)> : tensor<3xi64>
#poison_partial = #ub.poison<"DEADBEEF", arith(dense<1>)> : tensor<32xi64>
#poison_nested = #ub.poison<"DEAD0000", ub(#ub.poison<"BEEF", arith(dense<2>)>)> : tensor<32xi64>

// CHECK-LABEL: func.func @poison(
// CHECK-SAME: a0 = #ub.poison : bf16
// CHECK-SAME: a1 = #ub.poison<arith(dense<1>)> : tensor<3xi64>
// CHECK-SAME: a2 = #ub.poison<"00000000DEADBEEF", arith(dense<1>)> : tensor<32xi64>
// CHECK-SAME: a3 = #ub.poison<"00000000DEADBEEF", ub(dense<2>)> : tensor<32xi64>
func.func @poison() attributes { a0 = #poison_full, a1 = #poison_defined, a2 = #poison_partial, a3 = #poison_nested } {
    return
}
