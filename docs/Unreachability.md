# Unreachability

- [Unreachability](#unreachability)
  - [Data flow](#data-flow)
    - [Example](#example)
  - [Control flow](#control-flow)
    - [Blocks](#blocks)
      - [Example](#example-1)
    - [Regions](#regions)
    - [Functions](#functions)


**Unreachability** is an eagerly propagated but pessimistic IR proposition. It indicates that a value never becomes available, because control flow never returns from some operation.

Unreachability is encoded in the `ub` dialect by

* the discaradable attribute `ub.unreachable`,
* the type `!ub.never`,
* the attribute `#ub.never`,
* and the operation `ub.never`.

There are two distinct analyses that lead to unreachable code:

* [Data flow](#data-flow) propagation of `never` values.
* [Control flow](#control-flow) propagation of `ub.unreachable` terminators.

## Data flow

The `#ub.never` attribute is the compile-time proposition indicating that a value will never become available. It is materialized by the the `ub.never` operation. All values produced by `ub.never`, or of type `!ub.never`, are considered unreachable, also called `never` values.

An operation is considered unreachable if it can never be scheduled. Unreachable operands never become available, and so any operation that depends on unreachable operands is unreachable.

Counterintuitively, an unreachable operation may still produce a compile-time value via constant folding. However, a `#ub.never` constant does not produce additional knowledge. Thus, any unfoldable operation with one or more operands, all of which are `never`, will never produce a result. Such operations are replaced with `never` values, and propagate unreachability.

All unreachable operations should eventually be replaced by `ub.never`, such that `ub.never` becomes the only operation in the code that does not return control flow (see [control flow](#control-flow)).

### Example

Consider the following snippet:

```mlir
%never = ub.never : i64
%false = arith.constant false
%cst0  = arith.constant 0 : i64
%0     = arith.addi %never, %never : i64
%1     = arith.select %false, %never, %cst0 : i64
```

Here, `arith.addi` is unreachable, unfoldable, and depends on unreachable values exclusively. It is thus canonicalized to `ub.never : i64`. In contrast, `arith.select` is unreachable, but can be folded to `%cst0`. Both transformations combined result in:

```mlir
%0 = ub.never : i64
%1 = arith.constant 0 : i64
```

## Control flow

An operation does not return control flow if it leads to an exception or an infinite loop, i.e., never terminates. Operations that must be scheduled after operations that do not return are unreachable by definition.

All results of an operation that is not returned from are replaced with `never` values.

Control flow in MLIR is only well-defined within regions that have the SSACFG property, and operations that implement the `RegionBranchOpInterface`. Otherwise, no assumptions about control flow may be made.

### Blocks

The `ub.unreachable` attribute annotates a terminator that is known to be unreachable. This proposition indicates that the branch is never taken, and thus the conatining block will never return.

A block is unreachable if it is not an entry block, and all of its predecessors are terminated by unreachable terminators.

#### Example

Following the [data flow](#data-flow) argument, any branch that depends on never values is unreachable by definition:

```mlir
^bb0(%arg0: i64):
    %never = ub.never : i64
    cf.br ^bb1(%never: i64, %arg0: i64) {ub.unreachable}

// preds: ^bb0
^bb1(%arg1: i64):
    ...
```

In the above example, `^bb1` is unreachable.

### Regions

Operations implementing the `RegionBranchOpInterface` define additional control flow semantics for their contained regions.

A region is not returned from if all of its `RegionBranchTerminatorOpInterface` terminators are unreachable. The parent operation is not returned from if all of its exiting regions are not returned from.

A region is unreachable if all of its predecessors are not returned from. If the parent operation is a predecessor, it must be unreachable.

### Functions

Operations implementing the `FunctionOpInterface` define additional control flow semantics via the call graph.

A function does not return if all of its return operations are unreachable.
A call is not returned from if the function does not return.

All results of a call that is not returned from are replaced with `never` values. Functions that do not return may be rewritten to return the `never` type.
