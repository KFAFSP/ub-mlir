// RUN: ub-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{hex literal}}
#poison_not_hex = #ub.poison<"not_a_hex_literal", arith(1)> : si16

// -----

// expected-error@+1 {{keyword}}
#poison_source_dialect = #ub.poison<"00", (1)> : si16

// -----

// expected-error@+1 {{source dialect}}
#poison_source_dialect = #ub.poison<"00", unloaded_dialect(1)> : si16

// -----

func.func @unreachable_not_a_terminator() -> i64 {
    // expected-error@+1 {{terminators}}
    %cst0 = arith.constant {ub.unreachable} 0 : i64
    return %cst0 : i64
}

// -----

func.func @unreachable_not_SSACFG() {
    %cst0 = arith.constant 0 : i64
    test.graph_region {
        // expected-error@+1 {{control flow}}
        test.region_yield %cst0 : i64 {ub.unreachable}
    }
    return
}

