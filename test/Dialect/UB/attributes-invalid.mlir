// RUN: ub-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{hex string}}
#poison_not_hex = #ub.poison<"LEL", arith(1)> : si16

// -----

// expected-error@+1 {{keyword}}
#poison_source_dialect = #ub.poison<"00", (1)> : si16
