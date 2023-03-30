/// Implements the matching mechanism for operation folding.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <optional>
#include <tuple>
#include <type_traits>

// TODO: Move this into mlir-extras?
namespace mlir::ub::match {

//===----------------------------------------------------------------------===//
// IR constraints
//===----------------------------------------------------------------------===//

// clang-format off
/// Concept for an MLIR Type constraint.
template<class T>
concept TypeConstraint = std::is_base_of_v<Type, T>;
/// Concept for an MLIR Attribute constraint.
template<class T>
concept AttrConstraint = std::is_base_of_v<Attribute, T>;
/// Concept for an MLIR Op constraint.
template<class T>
concept OpConstraint = std::is_base_of_v<OpState, T>;
// clang-format on

//===----------------------------------------------------------------------===//
// Match
//===----------------------------------------------------------------------===//

/// Type that wraps a match result binding of type @p T .
template<class T>
struct Match : std::optional<std::tuple<T>> {
    using Base = std::optional<std::tuple<T>>;

    /*implicit*/ Match() : Base(std::nullopt) {}
    /*implicit*/ Match(std::nullopt_t) : Base(std::nullopt) {}
    /*implicit*/ Match(T t) : Base(std::in_place, std::forward<T>(t)) {}
    /*implicit*/ Match(const std::optional<T> &opt) : Base(std::nullopt)
    {
        if (opt) Base::emplace(opt.value());
    }
    /*implicit*/ Match(std::optional<T> &&opt) : Base(std::nullopt)
    {
        if (opt) Base::emplace(std::move(opt).value());
    }
    // clang-format off
    template<class U> requires (std::is_constructible_v<T, U>)
    /*implicit*/ Match(U&& u) : Base(std::in_place, std::forward<U>(u)) {}
    // clang-format on

    /// Unwraps the match result.
    ///
    /// @pre    `*this`
    const T &unwrap() const & { return std::get<0>(**this); }
    /// Unwraps the match result.
    ///
    /// @pre    `*this`
    T &&unwrap() && { return std::get<0>(std::move(**this)); }
};

/// Return value that indicates match failure.
static constexpr auto match_fail = std::nullopt;

//===----------------------------------------------------------------------===//
// MatcherBase
//===----------------------------------------------------------------------===//

/// Reserved index that indicates the end of the operands range.
static constexpr auto operands_end = std::size_t(-1);

/// Determines whether @p Arity operands can be consumed.
///
/// @pre    `op`
template<std::size_t Arity>
static constexpr bool
canConsumeOperands(Operation* op, std::size_t operandIndex)
{
    if constexpr (Arity == 0 || Arity == operands_end) {
        // Consumes no operands, or a variable amount, which is always
        // allowed.
        return true;
    } else {
        // Consumes 1 or more operands, which must be available.
        return (operandIndex + Arity) <= op->getNumOperands();
    }
}

/// Base class for a custom folding matcher.
///
/// A folding matcher translates the state of a folding operation, i.e., the
/// operation pointer and constant attribute range, into a Match.
struct MatcherBase {
    //===------------------------------------------------------------------===//
    // Exemplar
    //===------------------------------------------------------------------===//
    //
    // Derived classes must implement the following functionality.
    //

    /// Type of the result binding.
    using Binding = MatcherBase;
    /// Number of operands consumed by this matcher.
    ///
    /// If `arity == operands_end`, all remaining operands are consumed.
    static constexpr std::size_t arity = 0;

    /// Applies this matcher to the state of a folding operation.
    ///
    /// @pre    `op`
    /// @pre    `operands.size() == op->getNumOperands()`
    /// @pre    `canConsumeOperands<arity>(op, operandIndex)`
    [[nodiscard]] static Match<MatcherBase> match(
        Operation* op,
        ArrayRef<Attribute> operands,
        SmallVectorImpl<OpFoldResult> &result,
        std::size_t operandIndex)
    {
        std::ignore = std::tie(op, operands, result, operandIndex);
        return match_fail;
    }
};

// clang-format off
/// Concept for a matcher.
template<class T>
concept Matcher = std::is_base_of_v<MatcherBase, T>;
// clang-format on

//===----------------------------------------------------------------------===//
// OperandMatcher
//===----------------------------------------------------------------------===//

/// Base class for implementing single-operand matchers.
///
/// Provides an implementation of match() that delegates to overloads with
/// OpFoldResult, Value and Attribute .
template<class Derived, class T = Derived>
struct OperandMatcher : MatcherBase {
    /// Type of the result binding.
    using Binding = T;
    /// Number of operands consumed by this matcher.
    static constexpr std::size_t arity = 1;

    [[nodiscard]] static Match<Binding> match(
        Operation* op,
        ArrayRef<Attribute> operands,
        SmallVectorImpl<OpFoldResult> &,
        std::size_t operandIndex)
    {
        assert(op);
        assert(operandIndex < op->getNumOperands());
        assert(operandIndex < operands.size());

        if (operands[operandIndex])
            return Derived::match(OpFoldResult(operands[operandIndex]));
        else
            return Derived::match(OpFoldResult(op->getOperand(operandIndex)));
    }
    [[nodiscard]] static std::optional<Binding> match(OpFoldResult operand)
    {
        assert(operand);

        if (const auto attr = operand.dyn_cast<Attribute>())
            return Derived::match(attr);
        return Derived::match(operand.dyn_cast<Value>());
    }
    [[nodiscard]] static std::optional<Binding> match(Value)
    {
        return match_fail;
    }
    [[nodiscard]] static std::optional<Binding> match(Attribute)
    {
        return match_fail;
    }
};

/// Matches any operand as an OpFoldResult.
struct Any : OpFoldResult, OperandMatcher<Any> {
    using OperandMatcher::match;

    /*implicit*/ Any(OpFoldResult operand) : OpFoldResult(operand)
    {
        assert(operand);
    }

    [[nodiscard]] static Match<Any> match(OpFoldResult operand)
    {
        assert(operand);

        return Any(operand);
    }
};

/// Matches a typed value or typed attribute as an OpFoldResult.
template<TypeConstraint TypeConstraint = Type>
struct Typed : OpFoldResult, OperandMatcher<Typed<TypeConstraint>> {
    using OperandMatcher<Typed<TypeConstraint>>::match;

    /// Gets the type of @p operand .
    ///
    /// If @p operand is an attribute, it must be a TypedAttr, otherwise the
    /// returned type is nullptr.
    [[nodiscard]] static Type getType(OpFoldResult operand)
    {
        if (const auto value = operand.dyn_cast<Value>())
            return value.getType();
        if (const auto typed = llvm::dyn_cast_or_null<TypedAttr>(
                operand.dyn_cast<Attribute>()))
            return typed.getType();

        return {};
    }

    [[nodiscard]] static std::optional<Typed> match(OpFoldResult operand)
    {
        if (!llvm::isa<TypeConstraint>(getType(operand))) return match_fail;
        return Typed(operand);
    }

    explicit Typed(OpFoldResult operand) : OpFoldResult(operand)
    {
        assert(operand);
        assert(llvm::isa<TypeConstraint>(getType(operand)));
    }

    [[nodiscard]] TypeConstraint getType() const
    {
        return llvm::cast<TypeConstraint>(getType(*this));
    }
};

/// Matches a poisoned value as a PoisonAttr.
template<TypeConstraint TypeConstraint = Type>
struct Poisoned : PoisonAttr, OperandMatcher<Poisoned<TypeConstraint>> {
    using OperandMatcher<Poisoned<TypeConstraint>>::match;
    using Base = Typed<TypeConstraint>;

    [[nodiscard]] static std::optional<Poisoned> match(Attribute attr)
    {
        const auto poisonAttr = llvm::dyn_cast<PoisonAttr>(attr);
        if (!poisonAttr || !poisonAttr.isPoisoned()
            || !llvm::isa<TypeConstraint>(poisonAttr.getType()))
            return match_fail;
        return Poisoned(poisonAttr);
    }

    explicit Poisoned(PoisonAttr attr) : PoisonAttr(attr)
    {
        assert(attr);
        assert(attr.isPoisoned());
        assert(llvm::isa<TypeConstraint>(attr.getType()));
    }

    [[nodiscard]] TypeConstraint getType() const
    {
        return llvm::cast<TypeConstraint>(this->getType());
    }
};

/// Matches a fully poisoned value as a PoisonAttr.
template<TypeConstraint TypeConstraint = Type>
struct Poison : PoisonAttr, OperandMatcher<Poison<TypeConstraint>> {
    using OperandMatcher<Poison<TypeConstraint>>::match;

    [[nodiscard]] static std::optional<Poison> match(Attribute attr)
    {
        const auto match = Poisoned<TypeConstraint>::match(attr);
        if (!match || !match.unwrap().isPoison()) return match_fail;
        return Poison(match.unwrap());
    }

    explicit Poison(Poisoned<TypeConstraint> poisoned) : PoisonAttr(poisoned)
    {
        assert(poisoned);
        assert(poisoned.isPoison());
    }

    [[nodiscard]] TypeConstraint getType() const
    {
        return llvm::cast<TypeConstraint>(this->getType());
    }
};

/// Matches a non-poisoned value as an OpFoldResult.
template<TypeConstraint TypeConstraint = Type>
struct WellDefined : OpFoldResult, OperandMatcher<WellDefined<TypeConstraint>> {
    using OperandMatcher<WellDefined<TypeConstraint>>::match;
    using Base = Typed<TypeConstraint>;

    [[nodiscard]] static std::optional<WellDefined> match(OpFoldResult operand)
    {
        assert(operand);

        if (Poisoned<TypeConstraint>::match(operand)) return match_fail;
        return operand;
    }

    explicit WellDefined(OpFoldResult operand) : OpFoldResult(operand)
    {
        assert(operand);
        assert(Base::match(operand));
        assert(!Poisoned<TypeConstraint>::match(operand));
    }

    [[nodiscard]] TypeConstraint getType() const
    {
        return llvm::cast<TypeConstraint>(Base::getType(*this));
    }
};

//===----------------------------------------------------------------------===//
// BuiltinMatcher
//===----------------------------------------------------------------------===//

/// Trait class for implementing MatcherBase for @p T .
template<class T>
struct BuiltinMatcher {};

/// Allows custom matchers to participate.
template<Matcher T>
struct BuiltinMatcher<T> : T {};

/// Allows access to the MLIRContext.
template<>
struct BuiltinMatcher<MLIRContext*> : MatcherBase {
    [[nodiscard]] static Match<MLIRContext*> match(
        Operation* op,
        ArrayRef<Attribute>,
        SmallVectorImpl<OpFoldResult> &,
        std::size_t)
    {
        assert(op);

        return op->getContext();
    }
};

/// Allows access to the underlying operation.
template<>
struct BuiltinMatcher<Operation*> : MatcherBase {
    [[nodiscard]] static Match<Operation*> match(
        Operation* op,
        ArrayRef<Attribute>,
        SmallVectorImpl<OpFoldResult> &,
        std::size_t)
    {
        assert(op);

        return op;
    }
};

/// Allows matching the underlying operation.
template<OpConstraint OpConstraint>
struct BuiltinMatcher<OpConstraint> : MatcherBase {
    [[nodiscard]] static Match<OpConstraint> match(
        Operation* op,
        ArrayRef<Attribute>,
        SmallVectorImpl<OpFoldResult> &,
        std::size_t)
    {
        if (const auto constrained = llvm::dyn_cast<OpConstraint>(op))
            return constrained;
        return match_fail;
    }
};

/// Allows matching the remaining operand values.
template<>
struct BuiltinMatcher<ValueRange> : MatcherBase {
    static constexpr auto arity = operands_end;

    [[nodiscard]] static Match<ValueRange> match(
        Operation* op,
        ArrayRef<Attribute>,
        SmallVectorImpl<OpFoldResult> &,
        std::size_t operandIndex)
    {
        assert(op);

        if (operandIndex == operands_end) return ValueRange{};
        assert(operandIndex <= op->getNumOperands());
        return op->getOperands().drop_front(operandIndex);
    }
};

/// Allows matching the remaining operand types.
template<>
struct BuiltinMatcher<TypeRange> : MatcherBase {
    static constexpr auto arity = operands_end;

    [[nodiscard]] static Match<TypeRange> match(
        Operation* op,
        ArrayRef<Attribute>,
        SmallVectorImpl<OpFoldResult> &,
        std::size_t operandIndex)
    {
        assert(op);

        if (operandIndex == operands_end) return TypeRange{};
        assert(operandIndex <= op->getNumOperands());
        return TypeRange(op->getOperands().drop_front(operandIndex));
    }
};

/// Allows access to the remaining constant operand range.
template<>
struct BuiltinMatcher<ArrayRef<Attribute>> : MatcherBase {
    static constexpr auto arity = operands_end;

    [[nodiscard]] static Match<ArrayRef<Attribute>> match(
        Operation*,
        ArrayRef<Attribute> operands,
        SmallVectorImpl<OpFoldResult> &,
        std::size_t operandIndex)
    {
        if (operandIndex == operands_end) return ArrayRef<Attribute>{};
        assert(operandIndex <= operands.size());
        return operands.drop_front(operandIndex);
    }
};

/// Allows access to the result container.
template<>
struct BuiltinMatcher<SmallVectorImpl<OpFoldResult> &> : MatcherBase {
    [[nodiscard]] static Match<SmallVectorImpl<OpFoldResult> &> match(
        Operation*,
        ArrayRef<Attribute>,
        SmallVectorImpl<OpFoldResult> &result,
        std::size_t)
    {
        return result;
    }
};

//===----------------------------------------------------------------------===//
// BuiltinOperandMatcher
//===----------------------------------------------------------------------===//

template<class T>
using BuiltinOperandMatcher = OperandMatcher<BuiltinMatcher<T>, T>;

/// Allows matching OpFoldResult.
template<>
struct BuiltinMatcher<OpFoldResult> : BuiltinOperandMatcher<OpFoldResult> {
    using OperandMatcher::match;

    [[nodiscard]] static std::optional<OpFoldResult> match(OpFoldResult operand)
    {
        return operand;
    }
};

/// Allows matching Value.
template<>
struct BuiltinMatcher<Value> : BuiltinOperandMatcher<Value> {
    using OperandMatcher::match;

    [[nodiscard]] static std::optional<Value> match(Value value)
    {
        return value;
    }
};

/// Allows matching TypedValue.
template<TypeConstraint TypeConstraint>
struct BuiltinMatcher<mlir::detail::TypedValue<TypeConstraint>>
        : BuiltinOperandMatcher<mlir::detail::TypedValue<TypeConstraint>> {
    using OperandMatcher<mlir::detail::TypedValue<TypeConstraint>>::match;

    [[nodiscard]] static std::optional<TypedValue<TypeConstraint>>
    match(Value value)
    {
        if (!llvm::isa<TypeConstraint>(value.getType())) return match_fail;
        return TypedValue<TypeConstraint>(value);
    }
};

/// Allows matching TypeConstraint.
template<TypeConstraint TypeConstraint>
struct BuiltinMatcher<TypeConstraint>
        : OperandMatcher<BuiltinMatcher<TypeConstraint>, TypeConstraint> {
    using OperandMatcher<BuiltinMatcher<TypeConstraint>, TypeConstraint>::match;
    using Base = Typed<TypeConstraint>;

    [[nodiscard]] static std::optional<TypeConstraint>
    match(OpFoldResult operand)
    {
        const auto match = Base::match(operand);
        if (!match) return match_fail;
        return match.unwrap().getType();
    }
};

/// Allows matching AttrConstraint.
template<AttrConstraint AttrConstraint>
struct BuiltinMatcher<AttrConstraint>
        : OperandMatcher<BuiltinMatcher<AttrConstraint>, AttrConstraint> {
    using OperandMatcher<BuiltinMatcher<AttrConstraint>, AttrConstraint>::match;

    [[nodiscard]] static std::optional<AttrConstraint> match(Attribute attr)
    {
        if (const auto constrained = llvm::dyn_cast<AttrConstraint>(attr))
            return constrained;
        return match_fail;
    }
};

//===----------------------------------------------------------------------===//
// makePattern
//===----------------------------------------------------------------------===//

namespace detail {

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

/// Creates a pattern for @p Prototype .
///
/// @pre    ArgumentIndex <= llvm::function_traits<Prototype>::num_args
template<
    class Prototype,
    std::size_t ArgumentIndex = 0,
    std::size_t OperandIndex = 0>
[[nodiscard]] auto makePattern()
{
    using PrototypeTraits = llvm::function_traits<Prototype>;
    constexpr auto prototypeArity = PrototypeTraits::num_args;
    static_assert(
        ArgumentIndex <= prototypeArity,
        "ArgumentIndex out of range");
    using ResultType = detail::wrap_result_t<Prototype, ArgumentIndex>;

    if constexpr (ArgumentIndex == prototypeArity) {
        // All arguments have been visited, so we can finalize the super matcher
        // by checking that all operands have been consumed.
        return [](Operation* op, auto &&...) -> ResultType {
            if (OperandIndex < op->getNumOperands()) return std::nullopt;
            return std::tuple();
        };
    } else {
        // Get the matcher for this argument.
        using ArgumentType =
            typename PrototypeTraits::template arg_t<ArgumentIndex>;
        using MatcherType = BuiltinMatcher<ArgumentType>;
        static_assert(Matcher<MatcherType>, "Argument has no matcher");

        // Handle the consumption of operands by the matcher.
        static_assert(
            MatcherType::arity == 0 || MatcherType::arity == operands_end
                || OperandIndex != operands_end,
            "Ran out of operands to match");
        constexpr auto nextOperand = MatcherType::arity == operands_end
                                         ? operands_end
                                         : OperandIndex + MatcherType::arity;

        // Combine the matcher recursively.
        return [](Operation* op,
                  ArrayRef<Attribute> operands,
                  SmallVectorImpl<OpFoldResult> &result) -> ResultType {
            // Recurse into the tail through IIFE.
            if (auto tail =
                    makePattern<Prototype, ArgumentIndex + 1, nextOperand>()(
                        op,
                        operands,
                        result)) {
                // Check that the operands can be consumed.
                if (!canConsumeOperands<MatcherType::arity>(op, OperandIndex))
                    return match_fail;

                // Apply the matcher.
                if (auto match =
                        MatcherType::match(op, operands, result, OperandIndex))
                    return std::tuple_cat(std::move(*match), std::move(*tail));

                // TODO: Add some debugging here: match failed!
            }
            return match_fail;
        };
    }
}

} // namespace detail

/// Creates a pattern for @p Prototype .
template<class Prototype>
[[nodiscard]] auto makePattern()
{
    return detail::makePattern<Prototype>();
}

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

//===----------------------------------------------------------------------===//
// FoldCallback
//===----------------------------------------------------------------------===//

/// Signature of a fold callback for generic operations.
using FoldCallback = LogicalResult(
    Operation*,
    ArrayRef<Attribute>,
    SmallVectorImpl<OpFoldResult> &);

/// Owning reference to a FoldCallback invocable.
using Folder = std::function<match::FoldCallback>;

/// Makes a Folder from @p pattern .
[[nodiscard]] Folder makeFolder(auto &&pattern)
{
    using Prototype = std::decay_t<decltype(pattern)>;
    using PrototypeTraits = llvm::function_traits<Prototype>;
    static_assert(PrototypeTraits::num_args != 0, "Trivially empty pattern");

    return [head = std::forward<decltype(pattern)>(pattern)](
               Operation* op,
               ArrayRef<Attribute> operands,
               SmallVectorImpl<OpFoldResult> &result) -> LogicalResult {
        if (const auto match = makePattern<Prototype>()(op, operands, result))
            return makeCompleter<Prototype>()(
                op,
                operands,
                result,
                std::apply(head, *match));
        return failure();
    };
}

} // namespace mlir::ub::match
