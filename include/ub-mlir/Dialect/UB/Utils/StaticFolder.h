/// Implements a template helper for declarative operation folding.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/OpImplementation.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"

#include <array>
#include <optional>
#include <tuple>
#include <type_traits>

namespace mlir::ub {

namespace detail {

/// Type of the operation reference argument.
using OperationArg = Operation*;
/// Type of the constant operands range argument.
using OperandsArg = ArrayRef<Attribute>;
/// Type of the result container argument.
using ResultArg = SmallVectorImpl<OpFoldResult> &;

/// Canonical fold callback for generic operations.
using FoldCallback =
    function_ref<LogicalResult(OperationArg, OperandsArg, ResultArg)>;

/// Gets a tuple of @p Fn argument types at @p Indices .
template<class Fn, class Indices>
struct arg_tuple;
template<class Fn>
struct arg_tuple<Fn, std::index_sequence<>> {
    using type = std::tuple<>;
};
template<class Fn, std::size_t... Is>
struct arg_tuple<Fn, std::index_sequence<Is...>> {
    using type =
        std::tuple<typename llvm::function_traits<Fn>::template arg_t<Is>...>;
};
template<class Fn, class Indices>
using arg_tuple_t = typename arg_tuple<Fn, Indices>::type;

/// Adds @p I to @p Sequence .
template<std::size_t I, class Sequence>
struct shift_seq;
template<std::size_t I, std::size_t... Is>
struct shift_seq<I, std::index_sequence<Is...>> {
    using type = std::index_sequence<(I + Is)...>;
};
template<std::size_t I, class Indices>
using shift_seq_t = typename shift_seq<I, Indices>::type;

/// Obtains the argument indices into @p Fn starting from @p I0 .
///
/// @pre    I0 <= llvm::function_traits<Fn>::num_args
template<class Fn, std::size_t I0 = 0>
using arg_indices_t = shift_seq_t<
    I0,
    std::make_index_sequence<llvm::function_traits<Fn>::num_args - I0>>;

/// Gets the argument tuple of @p Fn starting with @p I0 .
///
/// @pre    I0 <= llvm::function_traits<Fn>::num_args
template<class Fn, std::size_t I0>
using wrap_tuple_t = arg_tuple_t<Fn, arg_indices_t<Fn, I0>>;
/// Gets the optional argument tuple of @p Fn starting with @p I0 .
///
/// @pre    I0 <= llvm::function_traits<Fn>::num_args
template<class Fn, std::size_t I0>
using wrap_result_t = std::optional<wrap_tuple_t<Fn, I0>>;

/// Determines whether @p T is a TypedValue, and what type it specializes.
template<class T>
struct is_typed_value : std::false_type {};
template<class Type>
struct is_typed_value<mlir::detail::TypedValue<Type>> : std::true_type {
    using type = Type;
};
template<class T>
static constexpr auto is_typed_value_v = is_typed_value<T>::value;
template<class T>
using is_typed_value_t = typename is_typed_value<T>::type;

/// Index reserved for the last operand.
static constexpr auto last_op_idx = std::size_t(-1);

/// Creates a tuple argument wrapper for @p Fn .
///
/// @pre    ArgIdx <= llvm::function_traits<Fn>::num_args
template<class Fn, std::size_t ArgIdx = 0, std::size_t OpIdx = 0>
[[nodiscard]] auto wrapArguments()
{
    using fn_traits = llvm::function_traits<Fn>;
    constexpr auto arity = fn_traits::num_args;
    using result_type = wrap_result_t<Fn, ArgIdx>;

    if constexpr (ArgIdx == arity) {
        // All arguments have been wrapped, finalize the callback by checking
        // that all operands have been consumed.
        return [](OperationArg op, auto &&...) -> result_type {
            if (OpIdx < op->getNumOperands()) return std::nullopt;
            return std::tuple();
        };
    } else {
        // Wrap one more argument and continue recursively.
        using arg_type = typename fn_traits::template arg_t<ArgIdx>;

        if constexpr (std::is_same_v<arg_type, OperationArg>) {
            // OperationArg
            // Gets the pointer to the operation.
            return [](OperationArg op,
                      OperandsArg operands,
                      ResultArg result) -> result_type {
                if (auto tail = wrapArguments<Fn, ArgIdx + 1, OpIdx>()(
                        op,
                        operands,
                        result))
                    return std::tuple_cat(std::tuple(op), std::move(*tail));
                return std::nullopt;
            };
        } else if constexpr (std::is_same_v<arg_type, OperandsArg>) {
            // OperandsArg
            // Gets the range of constant operands.
            return [](OperationArg op,
                      OperandsArg operands,
                      ResultArg result) -> result_type {
                if (auto tail = wrapArguments<Fn, ArgIdx + 1, OpIdx>()(
                        op,
                        operands,
                        result))
                    return std::tuple_cat(
                        std::tuple(operands),
                        std::move(*tail));
                return std::nullopt;
            };
        } else if constexpr (std::is_same_v<arg_type, ResultArg>) {
            // ResultArg
            // Gets the container for inserting fold results into.
            return [](OperationArg op,
                      OperandsArg operands,
                      ResultArg result) -> result_type {
                if (auto tail = wrapArguments<Fn, ArgIdx + 1, OpIdx>()(
                        op,
                        operands,
                        result))
                    return std::tuple_cat(
                        std::forward_as_tuple<ResultArg>(result),
                        std::move(*tail));
                return std::nullopt;
            };
        } else if constexpr (std::is_base_of_v<OpState, arg_type>) {
            // OpState
            // Matches an operation of a specific type and returns it.
            return [](OperationArg op,
                      OperandsArg operands,
                      ResultArg result) -> result_type {
                if (auto tail = wrapArguments<Fn, ArgIdx + 1, OpIdx>()(
                        op,
                        operands,
                        result)) {
                    if (const auto opArg = dyn_cast<arg_type>(op))
                        return std::tuple_cat(
                            std::tuple(opArg),
                            std::move(*tail));
                }
                return std::nullopt;
            };
        } else if constexpr (std::is_same_v<arg_type, TypeRange>) {
            // TypeRange
            // Consumes all remaining operands and returns their types.
            return [](OperationArg op,
                      OperandsArg operands,
                      ResultArg result) -> result_type {
                if (auto tail = wrapArguments<Fn, ArgIdx + 1, last_op_idx>()(
                        op,
                        operands,
                        result))
                    return std::tuple_cat(
                        std::tuple(
                            op->getOperands().drop_front(OpIdx).getTypes()),
                        std::move(*tail));
                return std::nullopt;
            };
        } else if constexpr (std::is_same_v<arg_type, ValueRange>) {
            // ValueRange
            // Consumes all remaining operands and returns their values.
            return [](OperationArg op,
                      OperandsArg operands,
                      ResultArg result) -> result_type {
                if (auto tail = wrapArguments<Fn, ArgIdx + 1, last_op_idx>()(
                        op,
                        operands,
                        result))
                    return std::tuple_cat(
                        std::tuple(op->getOperands().drop_front(OpIdx)),
                        std::move(*tail));
                return std::nullopt;
            };
        } else if constexpr (std::is_base_of_v<OpFoldResult, arg_type>) {
            static_assert(OpIdx != last_op_idx, "Trivially unmatchable folder");
            // OpFoldResult
            // Consumes the next operand.
            return [](OperationArg op,
                      OperandsArg operands,
                      ResultArg result) -> result_type {
                if (auto tail = wrapArguments<Fn, ArgIdx + 1, OpIdx + 1>()(
                        op,
                        operands,
                        result)) {
                    if (op->getNumOperands() <= OpIdx) return std::nullopt;
                    const auto opArg =
                        operands[OpIdx] ? OpFoldResult(operands[OpIdx])
                                        : OpFoldResult(op->getOperand(OpIdx));
                    return std::tuple_cat(std::tuple(opArg), std::move(*tail));
                }
                return std::nullopt;
            };
        } else if constexpr (is_typed_value_v<arg_type>) {
            static_assert(OpIdx != last_op_idx, "Trivially unmatchable folder");
            // TypedValue<T>
            // Consumes the next operand and matches its type to T.
            return [](OperationArg op,
                      OperandsArg operands,
                      ResultArg result) -> result_type {
                if (auto tail = wrapArguments<Fn, ArgIdx + 1, OpIdx + 1>()(
                        op,
                        operands,
                        result)) {
                    if (op->getNumOperands() <= OpIdx) return std::nullopt;
                    const auto opArg = op->getOperand(OpIdx);
                    if (!opArg.getType().isa<is_typed_value_t<arg_type>>())
                        return std::nullopt;
                    return std::tuple_cat(
                        std::tuple(arg_type(opArg)),
                        std::move(*tail));
                }
                return std::nullopt;
            };
        } else if constexpr (std::is_base_of_v<Value, arg_type>) {
            static_assert(OpIdx != last_op_idx, "Trivially unmatchable folder");
            // Value
            // Consumes the next operand.
            return [](OperationArg op,
                      OperandsArg operands,
                      ResultArg result) -> result_type {
                if (auto tail = wrapArguments<Fn, ArgIdx + 1, OpIdx + 1>()(
                        op,
                        operands,
                        result)) {
                    if (op->getNumOperands() <= OpIdx) return std::nullopt;
                    if constexpr (std::is_same_v<Value, arg_type>) {
                        return std::tuple_cat(
                            std::tuple(op->getOperand(OpIdx)),
                            std::move(*tail));
                    } else {
                        if (const auto opArg =
                                op->getOperand(OpIdx).dyn_cast<arg_type>())
                            return std::tuple_cat(
                                std::tuple(opArg),
                                std::move(*tail));
                    }
                }
                return std::nullopt;
            };
        } else if constexpr (std::is_base_of_v<Attribute, arg_type>) {
            static_assert(OpIdx != last_op_idx, "Trivially unmatchable folder");
            // Attribute
            // Consumes the next operand if it is constant.
            return [](OperationArg op,
                      OperandsArg operands,
                      ResultArg result) -> result_type {
                if (auto tail = wrapArguments<Fn, ArgIdx + 1, OpIdx + 1>()(
                        op,
                        operands,
                        result)) {
                    if (op->getNumOperands() <= OpIdx) return std::nullopt;
                    if constexpr (std::is_same_v<Attribute, arg_type>) {
                        if (const auto opArg = operands[OpIdx]) {
                            return std::tuple_cat(
                                std::tuple(opArg),
                                std::move(*tail));
                        }
                    } else {
                        if (const auto opArg =
                                operands[OpIdx].dyn_cast_or_null<arg_type>())
                            return std::tuple_cat(
                                std::tuple(opArg),
                                std::move(*tail));
                    }
                }
                return std::nullopt;
            };
        } else {
            assert(false && "trivially unmatchable folder");
            return [](auto &&...) -> result_type { return std::nullopt; };
        }
    }
}

/// Creates a fold result unwrapper for @p Fn .
template<class Fn>
[[nodiscard]] auto unwrapResult()
{
    using fn_traits = llvm::function_traits<Fn>;
    using result_type = typename fn_traits::result_t;

    constexpr auto isLogicalResult =
        std::is_convertible_v<LogicalResult, result_type>;
    constexpr auto isFoldResult =
        std::is_constructible_v<OpFoldResult, result_type>;
    static_assert(isLogicalResult || isFoldResult, "Invalid result type");

    if constexpr (isLogicalResult) {
        // LogicalResult
        // Pass through.
        return [](OperationArg,
                  OperandsArg,
                  ResultArg,
                  auto &&fnResult) -> LogicalResult {
            return static_cast<LogicalResult>(fnResult);
        };
    } else if constexpr (isFoldResult) {
        // Convertible to OpFoldResult
        // Push to results and indicate non-emptiness of OpFoldResult.
        return [](OperationArg,
                  OperandsArg,
                  ResultArg result,
                  auto &&fnResult) -> LogicalResult {
            return success(result.emplace_back(fnResult));
        };
    } else {
        llvm_unreachable("invalid result type");
        return [](auto &&...) -> LogicalResult { return failure(); };
    }
}

/// Wraps @p fn as a FoldCallback.
[[nodiscard]] FoldCallback wrapCallback(FoldCallback fn) { return fn; }
/// Wraps @p fn as a FoldCallback.
[[nodiscard]] FoldCallback wrapCallback(auto &&fn)
{
    using fn_type = std::decay_t<decltype(fn)>;
    using fn_traits = llvm::function_traits<fn_type>;
    static_assert(fn_traits::num_args != 0, "Trivially unmatched callback");

    return [fn = std::forward<decltype(fn)>(fn)](
               OperationArg op,
               OperandsArg operands,
               ResultArg result) -> LogicalResult {
        if (const auto match = wrapArguments<fn_type>()(op, operands, result))
            return unwrapResult<fn_type>()(
                op,
                operands,
                result,
                std::apply(fn, *match));
        return failure();
    };
}

} // namespace detail

/// Utility class that implements declarative folding implementations in C++.
template<std::size_t N>
class StaticFolder {
public:
    /// Initializes a StaticFolder from @p fns .
    /*implicit*/ StaticFolder(auto &&... fns)
            : m_callbacks{
                detail::wrapCallback(std::forward<decltype(fns)>(fns))...}
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
    /*implicit*/ operator detail::FoldCallback() const
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

    std::array<detail::FoldCallback, N> m_callbacks;
};

// CTAD for StaticFolder.
template<class... Fns>
StaticFolder(Fns &&...) -> StaticFolder<sizeof...(Fns)>;

} // namespace mlir::ub
