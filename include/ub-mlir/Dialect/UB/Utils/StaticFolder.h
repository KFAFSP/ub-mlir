/// Implements a template helper for declarative operation folding.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/OpImplementation.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"
#include "ub-mlir/Dialect/UB/Utils/Matchers.h"

#include <array>
#include <optional>
#include <tuple>
#include <type_traits>

namespace mlir::ub {

//===----------------------------------------------------------------------===//
// FoldCompleter
//===----------------------------------------------------------------------===//

/// Trait class for implementing folding complete actions for @p T .
template<class T>
struct FoldCompleter;

/// Allows using LogicalResult completion.
template<>
struct FoldCompleter<LogicalResult> {
    LogicalResult operator()(
        Operation* op,
        ArrayRef<Attribute>,
        SmallVectorImpl<OpFoldResult> &result,
        LogicalResult ok) const
    {
        assert(op);
        assert(result.size() == op->getNumResults());

        return ok;
    }
};

// clang-format off
/// Allows OpFoldResult completion for OneResult ops.
template<class T> requires (std::is_constructible_v<OpFoldResult, T>)
struct FoldCompleter<T> {
    LogicalResult operator()(
        Operation* op,
        ArrayRef<Attribute>,
        SmallVectorImpl<OpFoldResult> &result,
        T t) const
    {
        assert(op);
        assert(op->getNumResults() == 1);

        result.assign(1, OpFoldResult(t));
        return success(result.back());
    }
};
// clang-format on

/// Allows ArrayRef<OpFoldResult> completion.
template<>
struct FoldCompleter<ArrayRef<OpFoldResult>> {
    LogicalResult operator()(
        Operation* op,
        ArrayRef<Attribute>,
        SmallVectorImpl<OpFoldResult> &result,
        ArrayRef<OpFoldResult> copy) const
    {
        assert(op);
        if (copy.empty()) return failure();

        assert(copy.size() == op->getNumResults());
        result.assign(copy.begin(), copy.end());
        return success();
    }
};

namespace detail {

/// Creates the folding completer for @p Prototype .
///
/// @pre    Result type of @p Prototype has a FoldCompleter
template<class Prototype>
[[nodiscard]] auto makeCompleter()
{
    using ResultType = typename llvm::function_traits<Prototype>::result_t;
    using Completer = FoldCompleter<ResultType>;
    static_assert(
        requires { Completer{}; },
        "Result has no completer");

    return Completer{};
}

} // namespace detail

//===----------------------------------------------------------------------===//
// FoldCallback
//===----------------------------------------------------------------------===//

/// Canonical fold callback for generic operations.
using FoldCallback = function_ref<LogicalResult(
    Operation*,
    ArrayRef<Attribute>,
    SmallVectorImpl<OpFoldResult> &)>;

/// Returns @p callback .
[[nodiscard]] FoldCallback makeFolder(FoldCallback callback)
{
    return callback;
}
/// Instanciates @p matcher as a FoldCallback.
[[nodiscard]] FoldCallback makeFolder(auto &&matcher)
{
    using Prototype = std::decay_t<decltype(matcher)>;
    using PrototypeTraits = llvm::function_traits<Prototype>;
    static_assert(PrototypeTraits::num_args != 0, "Trivially empty matcher");

    // Make a lambda closure with IIFE contents.
    return [matcher = std::forward<decltype(matcher)>(matcher)](
               Operation* op,
               ArrayRef<Attribute> operands,
               SmallVectorImpl<OpFoldResult> &result) -> LogicalResult {
        if (const auto match =
                detail::combineMatchers<Prototype>()(op, operands, result))
            return detail::makeCompleter<Prototype>()(
                op,
                operands,
                result,
                std::apply(matcher, *match));
        return failure();
    };
}

//===----------------------------------------------------------------------===//
// StaticFolder
//===----------------------------------------------------------------------===//

/// Utility class that implements declarative op folding in C++.
template<std::size_t N>
class StaticFolder {
public:
    /// Initializes a StaticFolder from @p fns .
    /*implicit*/ StaticFolder(auto &&... fns)
            : m_callbacks{makeFolder(std::forward<decltype(fns)>(fns))...}
    {}

    /*implicit*/ StaticFolder(StaticFolder &&) = delete;
    StaticFolder &operator=(StaticFolder &&) = delete;

    /// Evaluates this StaticFolder.
    ///
    /// @pre    `op`
    /// @pre    `operands.size() == op->getNumOperands()`
    LogicalResult operator()(
        Operation* op,
        ArrayRef<Attribute> operands,
        SmallVectorImpl<OpFoldResult> &result) const
    {
        assert(op);
        assert(operands.size() == op->getNumOperands());
        result.reserve(op->getNumResults());
        result.clear();

        // Try all registered callbacks in order.
        return unroll(op, operands, result);
    }
    /// Evaluates this StaticFolder.
    ///
    /// @pre    `op`
    /// @pre    `op.getNumResults() == 1`
    /// @pre    `operands.size() == op->getNumOperands()`
    [[nodiscard]] OpFoldResult
    operator()(Operation* op, ArrayRef<Attribute> operands) const
    {
        assert(op);
        assert(op->getNumResults() == 1);

        // Delegate to the generic callback.
        SmallVector<OpFoldResult> result;
        if (succeeded((*this)(op, operands, result))) return result.front();
        return {};
    }

    /// Creates a FoldCallback for this StaticFolder.
    /*implicit*/ operator FoldCallback() const
    {
        return [&](Operation* op,
                   ArrayRef<Attribute> operands,
                   SmallVectorImpl<OpFoldResult> &result) -> LogicalResult {
            return (*this)(op, operands, result);
        };
    }

private:
    /// Unrolls the callback application.
    ///
    /// @pre    I <= N
    /// @pre    `op`
    /// @pre    `operands.size() == op->getNumOperands()`
    template<std::size_t I = 0>
    LogicalResult unroll(
        Operation* op,
        ArrayRef<Attribute> operands,
        SmallVectorImpl<OpFoldResult> &result) const
    {
        if constexpr (I == N) {
            return failure();
        } else {
            if (succeeded(m_callbacks[I](op, operands, result)))
                return success();
            return unroll<I + 1>(op, operands, result);
        }
    }

    std::array<FoldCallback, N> m_callbacks;
};

// CTAD for StaticFolder.
template<class... Fns>
StaticFolder(Fns &&...) -> StaticFolder<sizeof...(Fns)>;

} // namespace mlir::ub
