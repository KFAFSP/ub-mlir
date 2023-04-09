/// Declares the unreachability analysis.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Transforms/FoldUtils.h"
#include "ub-mlir/Dialect/UB/IR/Ops.h"
#include "ub-mlir/Dialect/UB/IR/Types.h"
#include "ub-mlir/Dialect/UB/Interfaces/ControlFlowTerminatorOpInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

namespace mlir::ub {

//===----------------------------------------------------------------------===//
// Concepts
//===----------------------------------------------------------------------===//

/// Concept for a Value that is known to be unreachable.
///
/// This concept is satisfied by:
///
///     - An OpResult from a NeverOp.
///     - Any value of NeverType.
///
/// Additionally, an UnreachableValue is considered to be typed if it is not of
/// the NeverType. In that case, its users are not in canonical form yet.
class UnreachableValue : public Value {
public:
    /// Determines whether @p value is an UnreachableValue.
    ///
    /// @pre    `value`
    [[nodiscard]] static bool classof(Value value)
    {
        return llvm::isa_and_present<NeverOp>(value.getDefiningOp())
               || llvm::isa<NeverType>(value.getType());
    }

    using Value::Value;

    /// Determines whether this value has an explicit type.
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isTyped() { return !llvm::isa<NeverType>(getType()); }
};

/// Concept for a ControlFlowTerminator that is known to be unreachable.
///
/// This concept is satisfied by:
///
///     - A ControlFlowTerminator that is known to be unreachable.
///
/// Additionally, this concept is buildable as an operation, in which case an
/// UnreachableOp is instanciated.
class UnreachableTerminator : public ControlFlowTerminator {
public:
    /// @copydoc classof(Operation*)
    [[nodiscard]] static bool classof(ControlFlowTerminator op)
    {
        return op.isKnownUnreachable();
    }
    /// Determines whether @p op is an UnreachableTerminator.
    ///
    /// @pre    `op`
    [[nodiscard]] static bool classof(Operation* op)
    {
        return llvm::TypeSwitch<Operation*, bool>(op)
            .Case([](ControlFlowTerminator op) { return classof(op); })
            .Default([](auto) { return false; });
    }

    /// Initializes a null UnreachableTerminator.
    explicit UnreachableTerminator() : ControlFlowTerminator(nullptr) {}
    /// Initializes a null UnreachableTerminator.
    /*implicit*/ UnreachableTerminator(std::nullptr_t)
            : ControlFlowTerminator(nullptr)
    {}
    /// Initializes an UnreachableTerminator from @p op.
    ///
    /// @pre    `llvm::isa<UnreachableTerminator>(op)`
    explicit UnreachableTerminator(Operation* op) : ControlFlowTerminator(op) {}

    /// Initializes an UnreachableTerminator from @p op .
    /*implicit*/ UnreachableTerminator(UnreachableOp op)
            : ControlFlowTerminator(op)
    {}

    /// Gets the UnreachableOp operation name.
    static constexpr llvm::StringLiteral getOperationName()
    {
        return UnreachableOp::getOperationName();
    }
    /// Builds an UnreachableOp.
    static void build(OpBuilder &builder, OperationState &state)
    {
        UnreachableOp::build(builder, state);
    }

    /// Returns @c true .
    [[nodiscard]] bool isKnownUnreachable() { return true; }
    /// Returns @c false .
    bool markAsUnreachable(RewriterBase &) { return false; }
};

//===----------------------------------------------------------------------===//
// Pessimistic unreachability tests
//===----------------------------------------------------------------------===//
//
// These functions test the reachability propositions `unreachable` and
// `noreturn` using shallow but inexpensive tests on the IR. More specifically,
// if the input IR is fully propagated and canonicalized, these functions will
// detect all applicable propositions.
//

/// Determines whether @p value is known to be unreachable.
///
/// See UnreachableValue for more details.
///
/// @pre    `value`
[[nodiscard]] inline bool isKnownUnreachable(Value value)
{
    return llvm::isa<UnreachableValue>(value);
}
/// Determines whether @p op is known to be unreachable.
///
/// If @p op is a UnreachableTerminator, returns @c true . Otherwise, @p op is
/// known to be unreachable if any of its operands are known to be unreachable.
///
/// @pre    `op`
[[nodiscard]] inline bool isKnownUnreachable(Operation* op)
{
    return llvm::TypeSwitch<Operation*, bool>(op)
        .Case([](UnreachableTerminator) { return true; })
        .Default([](Operation* op) {
            return llvm::any_of(
                op->getOperands(),
                static_cast<bool (*)(Value)>(&isKnownUnreachable));
        });
}

/// Determines whether @p op is known to not return.
///
/// @p op is known to be unreachable if it is a NeverOp, or returns any value of
/// NeverType.
///
/// @pre    `op`
[[nodiscard]] inline bool isKnownNoReturn(Operation* op)
{
    return llvm::TypeSwitch<Operation*, bool>(op)
        .Case([](NeverOp) { return true; })
        .Default([](Operation* op) {
            return llvm::any_of(
                op->getResultTypes(),
                [](Type type) { return llvm::isa<NeverType>(type); });
        });
}
/// Determines whether @p block is known to not return.
///
/// @p block is known to not return if it has an UnreachableTerminator.
///
/// @pre    `block`
[[nodiscard]] inline bool isKnownNoReturn(Block* block)
{
    return !block->empty() && llvm::isa<UnreachableTerminator>(&block->back());
}
/// Determines whether @p region is known to not return.
///
/// @p region is known to not return if it is the immediate child of a
/// RegionBranchOpInterface, and all of its RegionBranchTerminatorOpInterface
/// ops directly nested below are unreachable.
[[nodiscard]] inline bool isKnownNoReturn(Region* region)
{
    auto iface = llvm::dyn_cast<RegionBranchOpInterface>(region->getParentOp());
    if (!iface) return false;
    return llvm::all_of(
        region->getOps<RegionBranchTerminatorOpInterface>(),
        static_cast<bool (*)(Operation*)>(&isKnownUnreachable));
}

/// Determines whether @p block is known to be unreachable.
///
/// @p block is known to be unreachable if it is not an entry block, and all of
/// its predecessors are known to not return.
///
/// @pre    `block`
[[nodiscard]] inline bool isKnownUnreachable(Block* block)
{
    return !block->isEntryBlock()
           && llvm::all_of(
               block->getPredecessors(),
               static_cast<bool (*)(Block*)>(&isKnownNoReturn));
}

//===----------------------------------------------------------------------===//
// UnreachabilityAnalysis
//===----------------------------------------------------------------------===//

/// Recursive unreachability analysis.
///
/// This helper class extends the pessimistic unreachability checks to more
/// complex forms of control flow, and memoizes the results. It is also possible
/// to feed it with assumptions. Note that known propositions always evaluate to
/// @c true and are not memoized.
///
/// Control flow that is followed by this analysis includes:
///
///     - Intra-block SSACFG control flow.
///     - Inter-block branches.
///     - RegionBranchOpInterface control flow.
///     - CallOpInterface control flow.
///
/// This analysis is observable, and listeners may be used to intercept and
/// materialize the transitive results in IR.
class UnreachabilityAnalysis {
    using HandleT = const void*;
    using LookupT = llvm::DenseSet<HandleT>;
    using ListenerT = std::function<bool(HandleT, TypeID)>;
    using ListenersT = llvm::SmallVector<ListenerT, 3>;

public:
    /// Determines whether @p value is unreachable.
    ///
    /// Value reachability is not memoized and not observable.
    ///
    /// @pre    `value`
    bool isUnreachable(Value value)
    {
        return llvm::TypeSwitch<Value, bool>(value)
            .Case([](UnreachableValue) { return true; })
            .Case(
                [&](OpResult value) { return isUnreachable(value.getOwner()); })
            .Default([](auto) { return false; });
    }
    /// Determines whether @p op is unreachable.
    ///
    /// @pre    `op`
    bool isUnreachable(Operation* op)
    {
        return isKnownUnreachable(op) || isUnreachableImpl(op);
    }
    /// Determines whether @p block is unreachable.
    ///
    /// @pre    `block`
    bool isUnreachable(Block* block)
    {
        return isKnownUnreachable(block) || isUnreachableImpl(block);
    }
    /// Determines whether @p region is unreachable.
    ///
    /// @pre    `region`
    bool isUnreachable(Region* region) { return isUnreachableImpl(region); }

    /// Assumes that @p op is unreachable.
    ///
    /// Returns @c true if the assumption is new.
    bool assumeUnreachable(Operation* op)
    {
        return assumeImpl(m_unreachable, op);
    }
    /// Assumes that @p block is unreachable.
    ///
    /// Returns @c true if the assumption is new.
    bool assumeUnreachable(Block* block)
    {
        return assumeImpl(m_unreachable, block);
    }
    /// Assumes that @p region is unreachable.
    ///
    /// Returns @c true if the assumption is new.
    bool assumeUnreachable(Region* region)
    {
        return assumeImpl(m_unreachable, region);
    }

    /// Attaches a @p listener that is called when unreachable is deduced.
    ///
    /// The @p listener must accept an IR element pointer and return bool. If
    /// @p listener returns @c false , the element is not memoized.
    template<class Listener>
    void listenUnreachable(Listener &&listener)
    {
        using FnTraits = llvm::function_traits<Listener>;
        attachListener<typename FnTraits::template arg_t<0>>(
            m_unreachableListeners,
            std::forward<Listener>(listener));
    }

    /// Determines whether @p op does not return.
    ///
    /// @pre    `op`
    bool isNoReturn(Operation* op)
    {
        return isKnownNoReturn(op) || isNoReturnImpl(op);
    }
    /// Determines whether @p block does not return.
    ///
    /// @pre    `op`
    bool isNoReturn(Block* block)
    {
        return isKnownNoReturn(block) || isNoReturnImpl(block);
    }
    /// Determines whether @p region does not return.
    ///
    /// @pre    `op`
    bool isNoReturn(Region* region)
    {
        return isKnownNoReturn(region) || isNoReturnImpl(region);
    }

    /// Assumes that @p op does not return.
    ///
    /// Returns @c true if the assumption is new.
    bool assumeNoReturn(Operation* op) { return assumeImpl(m_noreturn, op); }
    /// Assumes that @p block does not return.
    ///
    /// Returns @c true if the assumption is new.
    bool assumeNoReturn(Block* block) { return assumeImpl(m_noreturn, block); }
    /// Assumes that @p region does not return.
    ///
    /// Returns @c true if the assumption is new.
    bool assumeNoReturn(Region* region)
    {
        return assumeImpl(m_noreturn, region);
    }

    /// Attaches a @p listener that is called when noreturn is deduced.
    ///
    /// The @p listener must accept an IR element pointer and return bool. If
    /// @p listener returns @c false , the element is not memoized.
    template<class Listener>
    void listenNoReturn(Listener &&listener)
    {
        using FnTraits = llvm::function_traits<Listener>;
        attachListener<typename FnTraits::template arg_t<0>>(
            m_noreturnListeners,
            std::forward<Listener>(listener));
    }

private:
    bool computeUnreachable(Operation* op);
    bool computeUnreachable(Block* block);
    bool computeUnreachable(Region* region);

    bool computeNoReturn(Operation* op);
    bool computeNoReturn(Block* block);
    bool computeNoReturn(Region* region);

    template<class T>
    void
    attachListener(ListenersT &listeners, std::invocable<T> auto &&listener)
    {
        listeners.push_back([listener = std::forward<decltype(listener)>(
                                 listener)](HandleT handle, TypeID type) {
            if (type == TypeID::get<T>()) return listener(T(handle));
            return true;
        });
    }
    bool notifyListenersImpl(ListenersT &listeners, HandleT handle, TypeID type)
    {
        return llvm::count_if(
                   listeners,
                   [&](ListenerT &listener) { return listener(handle, type); })
               != static_cast<long>(listeners.size());
    }

    bool isUnreachableImpl(auto* ptr)
    {
        if (m_unreachable.contains(HandleT(ptr))) return true;
        if (computeUnreachable(ptr)) {
            if (notifyListenersImpl(
                    m_unreachableListeners,
                    HandleT(ptr),
                    TypeID::get<decltype(ptr)>()))
                m_unreachable.insert(HandleT(ptr));
            return true;
        }

        return false;
    }
    bool isNoReturnImpl(auto* ptr)
    {
        if (m_noreturn.contains(HandleT(ptr))) return true;
        if (computeNoReturn(ptr)) {
            if (notifyListenersImpl(
                    m_noreturnListeners,
                    HandleT(ptr),
                    TypeID::get<decltype(ptr)>()))
                m_noreturn.insert(HandleT(ptr));
            return true;
        }

        return false;
    }

    bool assumeImpl(LookupT &lookup, auto* ptr)
    {
        return lookup.insert(HandleT(ptr)).second;
    }

    LookupT m_unreachable, m_noreturn;
    ListenersT m_unreachableListeners, m_noreturnListeners;
};

} // namespace mlir::ub
