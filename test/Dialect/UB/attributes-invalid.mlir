// RUN: ub-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{hex literal}}
#poison_not_hex = #ub.poison<"not_a_hex_literal", arith(1)> : si16

// -----

// expected-error@+1 {{keyword}}
#poison_source_dialect = #ub.poison<"00", (1)> : si16

// -----

// expected-error@+1 {{source dialect}}
#poison_source_dialect = #ub.poison<"00", unloaded_dialect(1)> : si16
