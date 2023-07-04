// RUN: ub-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{keyword}}
#poison_source_dialect = #ub.poisoned_elements<(1)[dense<1>]> : tensor<3xsi16>

// -----

// expected-error@+1 {{source dialect}}
#poison_source_dialect = #ub.poisoned_elements<unloaded_dialect(1)[dense<1>]> : tensor<3xsi16>
