// RUN: ub-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{expected valid keyword}}
#poison_source_dialect = #ubx.poisoned_elements<(1)[dense<1>]> : tensor<3xsi16>

// -----

// expected-error@+1 {{unknown source dialect}}
#poison_source_dialect = #ubx.poisoned_elements<unloaded_dialect(1)[dense<1>]> : tensor<3xsi16>
