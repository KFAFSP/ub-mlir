# Unreachability

- [Unreachability](#unreachability)
  - [Propositions](#propositions)
    - [Unreachable](#unreachable)
    - [Does not return](#does-not-return)
  - [Data flow](#data-flow)
    - [Example](#example)
  - [Control flow](#control-flow)
    - [Blocks](#blocks)
      - [Example](#example-1)
    - [Regions](#regions)
    - [Callables](#callables)
      - [Example](#example-2)


**Unreachability** is an eagerly propagated but pessimistic IR proposition.
It indicates that a value never becomes available, because control flow never returns from some operation.

Unreachability is encoded in the `ub` dialect by

- the discaradable attribute `ub.unreachable`,
- the operation `ub.unreachable`,
- the type `!ub.never`,
- the attribute `#ub.never`,
- and the operation `ub.never`.

There are two distinct propagation mechanisms for unreachable propositions:

- [Data flow](#data-flow) propagation of `never` values.
- [Control flow](#control-flow) propagation of `ub.unreachable` terminators.

## Propositions

The reachability analysis tracks two different propositions: `unreachable` and `noreturn`.

### Unreachable

The `unreachable` proposition indicates that an operation will never be executed at runtime.
To be precise, an unreachable operation is not scheduled.

An operation that is not scheduled will never produce its result values, thus preventing any dependent operations being scheduled.
Therefore, we extend `unreachable` to apply to values as well, indicating that they will never be available.

A block or region is `unreachable` if all of its predecessors are `noreturn`.

The `unreachable` proposition is encoded in the `ub` dialect as:

- The `ub.unreachable` SSACFG terminator, which may never be reached.
- The `ub.unreachable` discardable attribute, which applies to other terminators.
- The `ub.never` operation, which materializes an unreachable value.

### Does not return

The `noreturn` proposition indicates that control flow never returns from an operation.
A `noreturn` operation will never produce any result values, i.e., has `unreachable` results.

A block is `noreturn` if its terminator is `unreachable`.
A region is `noreturn` if all of its exit terminators are `unreachable`.
An operation defining region-based control flow is `noreturn` if all of its exiting regions are `noreturn`.

The `noreturn` proposition is encoded in the `ub` dialect as:

- The `!ub.never` type, which is the result type of a `noreturn` operation.
- The `ub.never` operation, which is the `noreturn` primitive.

## Data flow

The `#ub.never` attribute is the compile-time proposition indicating that a value will never become available.
It is materialized by the the `ub.never` operation.
All values produced by `ub.never`, or of type `!ub.never`, are considered unreachable, also called `never` values.

An operation that has a `never` operand is unreachable, and may never be scheduled.
It may never produce results at runtime.

Counterintuitively, an unreachable operation may still produce results at compile-time, via folding.
However, a `#ub.never` constant does not produce additional knowledge.
Thus, any unreachable operation with one or more operands, all of which are constant, will never produce results.
Such operations must always propagate `unreachable`.

### Example

Consider the following snippet:

```mlir
%never = ub.never : i64
%false = arith.constant false
%cst0  = arith.constant 0 : i64
%0     = arith.addi %cst0, %never : i64
%1     = arith.select %false, %never, %cst0 : i64
```

Here, `arith.addi` is unreachable, unfoldable, and depends on constants exclusively.
It is thus canonicalized to `ub.never : i64`.
In contrast, `arith.select` is unreachable, but can be folded to `%cst0`.
Both transformations combined result in:

```mlir
%0 = ub.never : i64
%1 = arith.constant 0 : i64
```

## Control flow

An operation that performs an infinite loop, results in an exception or otherwise prevents control flow from returning is `noreturn`.
All results of a `noreturn` operation are `unreachable`.

Control flow in MLIR is only well-defined in three contexts:

- Inside SSACFG regions, where control is transfered via `BranchOpInterface` ops.
- Inside and around `RegionBranchOpInterface` ops, where control flow is implicit between the surrounding and contained regions.
- Around `CallOpInterface` and `CallableOpInterface` ops, where control is transfered using calls and returns.

### Blocks

The `ub.unreachable` operation indicates a point in an SSACFG which may never be reached.
The `ub.unreachable` attribute indicates the same for any terminator in a well-defined control flow context.

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

In this example, `^bb0` is `noreturn`, and therefore `^bb1` becomes unreachable by definition.

### Regions

In an operation implementing the `RegionBranchOpInterface`, contained regions have well-defined control flow predecessors and successors.

A region is `noreturn` if all contained `RegionBranchTerminatorOpInterface` terminators are unreachable.
A region is `unreachable` if all of its predecessors are `noreturn`, and if the parent is a predecessor, it is `unreachable`.
The parent op is `noreturn` if all predecessor regions are `noreturn`.

### Callables

Operations implementing the `CallableOpInterface` are marked as `noreturn` by returning a result of `!ub.never` type.

A callable is `noreturn` if its region is `noreturn`.
Its region is `noreturn` if all of its return-like operators are `unreachable`.

Operations implementing the `CallOpInterface` are `noreturn` if their callee is `noreturn`.


#### Example

Consider a function that is known to not return:

```mlir
func.func @noreturn() -> i64 {
    %never = ub.never : i64
    return %never : i64
}

%0 = func.call @noreturn() : () -> i64
```

This information can be concretized in the IR by rewriting the result type of the function to `!ub.never`.
Regardless, the values returned by calls to this function are replaced by `never` values:

```mlir
func.func @noreturn() -> !ub.never {
    %never = ub.never
    return %never : !ub.never
}

func.call @noreturn() : () -> !ub.never
%0 = ub.never : i64
```
