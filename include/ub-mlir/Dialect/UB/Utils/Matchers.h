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

namespace mlir::ub {

//===----------------------------------------------------------------------===//
// MatcherBase
//===----------------------------------------------------------------------===//

/// Reserved index that indicates the end of the operands range.
static constexpr auto operands_end = std::size_t(-1);

/// Type that represents the result of @p Matcher .
template<class Matcher>
using Match = std::optional<std::tuple<Matcher>>;

/// Base class for a custom folding matcher.
///
/// A folding matcher translates the state of a folding operation, i.e., the
/// operation pointer and constant attribute range, into a Match.
struct MatcherBase {
    /// Type of the result binding.
    using Binding = MatcherBase;
    /// Number of operands consumed by this matcher.
    ///
    /// If `arity == operands_end`, all remaining operands are consumed.
    static constexpr std::size_t arity = 0;

    /// Creates a Match from @p binding .
    template<class T>
    [[nodiscard]] static Match<T> bind(std::optional<T> binding)
    {
        if (binding) return std::tuple(*binding);
        return std::nullopt;
    }

    /// Applies this matcher to the state of a folding operation.
    [[nodiscard]] static Match<MatcherBase> match(
        Operation*,
        ArrayRef<Attribute>,
        SmallVectorImpl<OpFoldResult> &,
        std::size_t)
    {
        return std::nullopt;
    }
};

/// Concept for a matcher.
template<class T>
concept Matcher = std::is_base_of_v<MatcherBase, T>;
/// Concept for an MLIR Type constraint.
template<class T>
concept TypeConstraint = std::is_base_of_v<Type, T>;
/// Concept for an MLIR Attribute constraint.
template<class T>
concept AttrConstraint = std::is_base_of_v<Attribute, T>;
/// Concept for an MLIR Op constraint.
template<class T>
concept OpConstraint = std::is_base_of_v<OpState, T>;

//===----------------------------------------------------------------------===//
// OperandMatcher
//===----------------------------------------------------------------------===//

/// Base class for implementing single-operand matchers.
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
            return bind(Derived::match(OpFoldResult(operands[operandIndex])));
        else
            return bind(
                Derived::match(OpFoldResult(op->getOperand(operandIndex))));
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
        return std::nullopt;
    }
    [[nodiscard]] static std::optional<Binding> match(Attribute)
    {
        return std::nullopt;
    }
};

namespace match {

/// Matches any operand as an OpFoldResult.
struct Any : OpFoldResult, OperandMatcher<Any> {
    using OperandMatcher::match;

    /*implicit*/ Any(OpFoldResult operand) : OpFoldResult(operand)
    {
        assert(operand);
    }

    [[nodiscard]] static Match<Any> match(OpFoldResult operand)
    {
        return Any(operand);
    }
};

/// Matches a typed value or attribute as an OpFoldResult.
template<TypeConstraint TypeConstraint = Type>
struct Typed : OpFoldResult, OperandMatcher<Typed<TypeConstraint>> {
    using OperandMatcher<Typed<TypeConstraint>>::match;

    [[nodiscard]] static Type getType(OpFoldResult operand)
    {
        if (const auto value = operand.dyn_cast<Value>())
            return value.getType();
        if (const auto typed =
                operand.dyn_cast<Attribute>().dyn_cast_or_null<TypedAttr>())
            return typed.getType();
        return {};
    }
    [[nodiscard]] static TypeConstraint getType(Type type)
    {
        if constexpr (std::is_same_v<Type, TypeConstraint>)
            return type;
        else
            return type.template cast<TypeConstraint>();
    }

    [[nodiscard]] static bool match(Type type)
    {
        if constexpr (std::is_same_v<Type, TypeConstraint>)
            return true;
        else
            return type.isa<TypeConstraint>();
    }
    [[nodiscard]] static std::optional<Typed> match(Attribute attr)
    {
        const auto typedAttr = attr.dyn_cast<TypedAttr>();
        if (!typedAttr) return std::nullopt;
        return Typed(attr);
    }
    [[nodiscard]] static std::optional<Typed> match(Value value)
    {
        if (!match(value)) return std::nullopt;
        return Typed(value);
    }

    explicit Typed(OpFoldResult operand) : OpFoldResult(operand)
    {
        assert(operand);
        assert(match(getType(operand)));
    }

    [[nodiscard]] TypeConstraint getType() const
    {
        return getType(getType(*this));
    }
};

/// Matches a fully poisoned value as a PoisonAttr.
template<TypeConstraint TypeConstraint = Type>
struct Poison : PoisonAttr, OperandMatcher<Poison<TypeConstraint>> {
    using OperandMatcher<Poison<TypeConstraint>>::match;
    using Base = Typed<TypeConstraint>;

    [[nodiscard]] static std::optional<Poison> match(Attribute attr)
    {
        assert(attr);

        const auto poisonAttr = attr.dyn_cast<PoisonAttr>();
        if (!poisonAttr || !poisonAttr.isPoison()) return std::nullopt;

        return Poison(poisonAttr);
    }

    explicit Poison(PoisonAttr attr) : PoisonAttr(attr)
    {
        assert(attr);
        assert(attr.isPoison());
        assert(Base::match(attr.getType()));
    }

    [[nodiscard]] TypeConstraint getType() const
    {
        return Base::getType(Base::getType(*this));
    }
};

/// Matches a poisoned value as a PoisonAttr.
template<TypeConstraint TypeConstraint = Type>
struct Poisoned : PoisonAttr, OperandMatcher<Poisoned<TypeConstraint>> {
    using OperandMatcher<Poisoned<TypeConstraint>>::match;
    using Base = Typed<TypeConstraint>;

    [[nodiscard]] static std::optional<Poisoned> match(Attribute attr)
    {
        assert(attr);

        const auto poisonAttr = attr.dyn_cast<PoisonAttr>();
        if (!poisonAttr || !poisonAttr.isPoisoned()) return std::nullopt;

        return Poisoned(poisonAttr);
    }

    explicit Poisoned(PoisonAttr attr) : PoisonAttr(attr)
    {
        assert(attr);
        assert(attr.isPoisoned());
        assert(Base::match(attr.getType()));
    }

    [[nodiscard]] TypeConstraint getType() const
    {
        return Base::getType(Base::getType(*this));
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

        if (Poisoned<TypeConstraint>::match(operand)) return std::nullopt;
        return operand;
    }

    explicit WellDefined(OpFoldResult operand) : OpFoldResult(operand)
    {
        assert(operand);
        assert(!Poisoned<TypeConstraint>::match(operand));
        assert(Typed<TypeConstraint>::match(operand));
    }

    [[nodiscard]] TypeConstraint getType() const
    {
        return Base::getType(Base::getType(*this));
    }
};

} // namespace match

//===----------------------------------------------------------------------===//
// BuiltinMatcher
//===----------------------------------------------------------------------===//

/// Trait class for implementing MatcherBase for @p T .
template<class T>
struct BuiltinMatcher {};

/// Allows custom matchers to participate.
template<Matcher T>
struct BuiltinMatcher<T> : T {};

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

        return std::tuple(op);
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
        assert(op);

        if (const auto constrained = llvm::dyn_cast<OpConstraint>(op))
            return std::tuple(constrained);
        return std::nullopt;
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

        if (operandIndex == operands_end) return std::tuple(ValueRange{});
        assert(operandIndex <= op->getNumOperands());
        return std::tuple(op->getOperands().drop_front(operandIndex));
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

        if (operandIndex == operands_end) return std::tuple(TypeRange{});
        assert(operandIndex <= op->getNumOperands());
        return std::tuple(
            TypeRange(op->getOperands().drop_front(operandIndex)));
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
        if (operandIndex == operands_end)
            return std::tuple(ArrayRef<Attribute>{});
        assert(operandIndex <= operands.size());
        return std::tuple(operands.drop_front(operandIndex));
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
        return std::tuple<decltype(result)>(result);
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
        assert(operand);

        return operand;
    }
};

/// Allows matching Value.
template<>
struct BuiltinMatcher<Value> : BuiltinOperandMatcher<Value> {
    using OperandMatcher::match;

    [[nodiscard]] static std::optional<Value> match(Value value)
    {
        assert(value);

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
        assert(value);

        if (!value.getType().isa<TypeConstraint>()) return std::nullopt;
        return TypedValue<TypeConstraint>(value);
    }
};

/// Allows matching TypeConstraint.
template<TypeConstraint TypeConstraint>
struct BuiltinMatcher<TypeConstraint>
        : OperandMatcher<BuiltinMatcher<TypeConstraint>, TypeConstraint> {
    using OperandMatcher<BuiltinMatcher<TypeConstraint>, TypeConstraint>::match;
    using Base = match::Typed<TypeConstraint>;

    [[nodiscard]] static std::optional<TypeConstraint>
    match(OpFoldResult operand)
    {
        assert(operand);

        const auto type = Base::getType(operand);
        if (Base::match(type)) return std::nullopt;
        return Base::getType(type);
    }
};

/// Allows matching AttrConstraint.
template<AttrConstraint AttrConstraint>
struct BuiltinMatcher<AttrConstraint>
        : OperandMatcher<BuiltinMatcher<AttrConstraint>, AttrConstraint> {
    using OperandMatcher<BuiltinMatcher<AttrConstraint>, AttrConstraint>::match;

    [[nodiscard]] static std::optional<AttrConstraint> match(Attribute attr)
    {
        assert(attr);

        if constexpr (std::is_same_v<Attribute, AttrConstraint>)
            return attr;
        else {
            const auto constrained = attr.dyn_cast<AttrConstraint>();
            if (!constrained) return std::nullopt;
            return constrained;
        }
    }
};

//===----------------------------------------------------------------------===//
// combineMatchers
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

/// Creates a combined matcher for @p Prototype .
///
/// @pre    ArgumentIndex <= llvm::function_traits<Prototype>::num_args
template<
    class Prototype,
    std::size_t ArgumentIndex = 0,
    std::size_t OperandIndex = 0>
[[nodiscard]] auto combineMatchers()
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
            if (auto tail = combineMatchers<
                    Prototype,
                    ArgumentIndex + 1,
                    nextOperand>()(op, operands, result)) {
                // Check that the operands can be consumed.
                if constexpr (
                    MatcherType::arity != 0
                    && MatcherType::arity != operands_end) {
                    if (OperandIndex + MatcherType::arity
                        > op->getNumOperands())
                        return std::nullopt;
                }

                // Apply the matcher.
                if (auto match =
                        MatcherType::match(op, operands, result, OperandIndex))
                    return std::tuple_cat(std::move(*match), std::move(*tail));

                // TODO: Add some debugging here: match failed!
            }
            return std::nullopt;
        };
    }
}

} // namespace detail

} // namespace mlir::ub
