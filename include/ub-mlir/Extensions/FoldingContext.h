/// Implements a matching mechanism for operation folding.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/OpImplementation.h"
#include "ub-mlir/Extensions/Concepts.h"
#include "ub-mlir/Extensions/FoldedRange.h"
#include "ub-mlir/Extensions/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <optional>
#include <tuple>
#include <type_traits>

namespace mlir::ext {

using match::Match;
using match::match_fail;

/// Type that stores the matching state.
template<std::size_t OperandIdx>
struct State {
    /*implicit*/ constexpr operator std::size_t() const { return OperandIdx; }
};

/// Reserved index that indicates the end of the operands range.
static constexpr State<std::size_t(-1)> operands_end{};

template<OpConstraint Op>
struct FoldingContext;

//===----------------------------------------------------------------------===//
// MatcherBase
//===----------------------------------------------------------------------===//

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

    /// Applies this matcher to a FoldingContext.
    template<OpConstraint Op>
    [[nodiscard]] static Match<Binding>
    match(const FoldingContext<Op> &ctx, std::size_t operandIdx);
};

/// Concept for a matcher.
template<class T>
concept Matcher = std::is_base_of_v<MatcherBase, T>;

//===----------------------------------------------------------------------===//
// BuiltinMatcher
//===----------------------------------------------------------------------===//

/// Trait class for implementing MatcherBase for @p T .
template<class T>
struct BuiltinMatcher : T {};

//===----------------------------------------------------------------------===//
// FoldingContext
//===----------------------------------------------------------------------===//

template<OpConstraint Op>
struct FoldingContext : match::ContextBase<LogicalResult>,
                        Op::template GenericAdaptor<FoldedRange> {
    using Base = typename Op::template GenericAdaptor<FoldedRange>;

    /// Initializes a FoldingContext by forwarding.
    ///
    /// @pre    `ctx`
    ///
    /// @post   `result.empty()`
    explicit FoldingContext(
        MLIRContext *ctx,
        SmallVectorImpl<OpFoldResult> &result,
        auto &&...args)
            : Base(std::forward<decltype(args)>(args)...),
              m_context(ctx),
              m_result(result)
    {
        assert(ctx);
    }

    /// Gets the underlying MLIRContext.
    MLIRContext *getContext() const { return m_context; }
    /// Gets the result container.
    SmallVectorImpl<OpFoldResult> &getResult() const { return m_result; }

    /// Gets the folded operand range.
    FoldedRange getOperands() const
    {
        // NOTE: MLIR IR objects are non-const, but contexts always are.
        return static_cast<Base &>(const_cast<FoldingContext &>(*this))
            .getOperands();
    }

    //===------------------------------------------------------------------===//
    // ContextBase
    //===------------------------------------------------------------------===//

    [[nodiscard]] constexpr State<0> init_state() const { return {}; }

    [[nodiscard]] constexpr bool assemble(auto operandIdx) const
    {
        return operandIdx >= this->getOperands().size();
    }

    /// Determines whether @p Arity operands can be consumed.
    template<std::size_t Arity>
    [[nodiscard]] constexpr auto consumeOperands(auto operandIdx) const
    {
        if constexpr (Arity == 0) {
            // Consumes no operands, which is always allowed.
            return std::make_pair(true, operandIdx);
        } else if constexpr (Arity == operands_end) {
            // Consumes the remaining operands, which is always allowed.
            return std::make_pair(true, operands_end);
        } else {
            // Consumes 1 or more operands, which must be available.
            constexpr State<operandIdx + Arity> nextIdx{};
            return std::make_pair(
                nextIdx <= this->getOperands().size(),
                nextIdx);
        }
    }

    template<class T>
    constexpr auto match(auto operandIdx) const
    {
        using MatcherType = BuiltinMatcher<T>;
        static_assert(Matcher<MatcherType>, "Expected Matcher");

        const auto [ok, newState] =
            consumeOperands<MatcherType::arity>(operandIdx);
        if (!ok) return std::make_tuple(Match<T>{}, newState);
        return std::make_tuple(MatcherType::match(*this, operandIdx), newState);
    }

    [[nodiscard]] match::PatternResult<LogicalResult>
    complete(auto, LogicalResult patternResult) const
    {
        if (succeeded(patternResult)) {
            assert(llvm::all_of(getResult(), [](auto x) { return !!x; }));
            return success();
        }

        return match_fail;
    }
    [[nodiscard]] match::PatternResult<LogicalResult>
    complete(auto, OpFoldResult patternResult) const
    {
        if (patternResult) {
            getResult().assign(1, patternResult);
            return success();
        }

        return match_fail;
    }

private:
    MLIRContext *m_context;
    SmallVectorImpl<OpFoldResult> &m_result;
};

/// Creates a FoldingContext for @p op using @p operands into @p result .
template<OpConstraint Op>
FoldingContext<Op> make_folding_context(
    Op op,
    ArrayRef<Attribute> operands,
    SmallVectorImpl<OpFoldResult> &result)
{
    result.reserve(op->getNumResults());
    result.clear();

    return FoldingContext<Op>(
        op->getContext(),
        result,
        FoldedRange(op.getOperands(), operands),
        op->getAttrDictionary(),
        op->getPropertiesStorage(),
        op->getRegions());
}

//===----------------------------------------------------------------------===//
// BuiltinMatcher
//===----------------------------------------------------------------------===//

/// Allows access to the MLIRContext.
template<>
struct BuiltinMatcher<MLIRContext *> : MatcherBase {
    [[nodiscard]] static Match<MLIRContext *>
    match(const auto &ctx, std::size_t)
    {
        return ctx.getContext();
    }
};

/// Allows access to the remaining folded operand range.
template<>
struct BuiltinMatcher<FoldedRange> : MatcherBase {
    static constexpr auto arity = operands_end;

    [[nodiscard]] static Match<FoldedRange>
    match(const auto &ctx, std::size_t operandIdx)
    {
        if (operandIdx == operands_end) return FoldedRange{};
        return ctx.getOperands().drop_front(operandIdx);
    }
};

/// Allows access to the remaining variable operand range.
template<>
struct BuiltinMatcher<ValueRange> : MatcherBase {
    static constexpr auto arity = operands_end;

    [[nodiscard]] static Match<ValueRange>
    match(const auto &ctx, std::size_t operandIdx)
    {
        if (operandIdx == operands_end) return ValueRange{};
        return ctx.getOperands().drop_front(operandIdx);
    }
};

/// Allows access to the remaining operand type range.
template<>
struct BuiltinMatcher<TypeRange> : MatcherBase {
    static constexpr auto arity = operands_end;

    [[nodiscard]] static Match<TypeRange>
    match(const auto &ctx, std::size_t operandIdx)
    {
        if (operandIdx == operands_end) return TypeRange{};
        return ctx.getOperands().drop_front(operandIdx).getTypes();
    }
};

/// Allows access to the remaining constant operand range.
template<>
struct BuiltinMatcher<ArrayRef<Attribute>> : MatcherBase {
    static constexpr auto arity = operands_end;

    [[nodiscard]] static Match<ArrayRef<Attribute>>
    match(const auto &ctx, std::size_t operandIdx)
    {
        if (operandIdx == operands_end) return ArrayRef<Attribute>{};
        return ctx.getOperands().drop_front(operandIdx);
    }
};

/// Allows access to the result container.
template<>
struct BuiltinMatcher<SmallVectorImpl<OpFoldResult> &> : MatcherBase {
    [[nodiscard]] static Match<SmallVectorImpl<OpFoldResult> &>
    match(const auto &ctx, std::size_t)
    {
        return ctx.getResult();
    }
};

/// Matches any operand as an OpFoldResult.
template<>
struct BuiltinMatcher<OpFoldResult> : MatcherBase {
    static constexpr auto arity = 1;

    [[nodiscard]] static Match<OpFoldResult>
    match(const auto &ctx, std::size_t operandIdx)
    {
        return ctx.getOperand(operandIdx);
    }
};

/// Matches an operand as a Value.
template<ValueConstraint AsValue>
struct BuiltinMatcher<AsValue> : MatcherBase {
    static constexpr auto arity = 1;

    [[nodiscard]] static Match<AsValue>
    match(const auto &ctx, std::size_t operandIdx)
    {
        if (auto value = llvm::dyn_cast<AsValue>(
                ctx.getOperands().getValues()[operandIdx]))
            return value;
        return match_fail;
    }
};

/// Matches an operand value as its Type.
template<TypeConstraint AsType>
struct BuiltinMatcher<AsType> : MatcherBase {
    static constexpr auto arity = 1;

    [[nodiscard]] static Match<AsType>
    match(const auto &ctx, std::size_t operandIdx)
    {
        if (auto type = llvm::dyn_cast<AsType>(
                ctx.getOperands().getValues()[operandIdx].getType()))
            return type;
        return match_fail;
    }
};

/// Matches an operand as an Attribute.
template<AttrConstraint AsAttr>
struct BuiltinMatcher<AsAttr> : MatcherBase {
    static constexpr auto arity = 1;

    [[nodiscard]] static Match<AsAttr>
    match(const auto &ctx, std::size_t operandIdx)
    {
        if (auto attr = llvm::dyn_cast_if_present<AsAttr>(
                ctx.getOperands().getAttributes()[operandIdx]))
            return attr;
        return match_fail;
    }
};

/// Matches an operand as a result of a constrained Op.
template<OpConstraint AsOp>
struct BuiltinMatcher<AsOp> : MatcherBase {
    static constexpr auto arity = 1;

    [[nodiscard]] static Match<AsOp>
    match(const auto &ctx, std::size_t operandIdx)
    {
        auto value = ctx.getOperands().getValues()[operandIdx];
        if (auto op = value.template getDefiningOp<AsOp>(value)) return op;
        return match_fail;
    }
};

} // namespace mlir::ext
