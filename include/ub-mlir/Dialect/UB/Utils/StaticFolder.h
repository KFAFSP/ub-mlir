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
// StaticFolder
//===----------------------------------------------------------------------===//

/// Utility class that implements declarative op folding in C++.
template<std::size_t N>
class StaticFolder : std::array<match::Folder, N> {
    using Base = std::array<match::Folder, N>;

    template<std::size_t M>
    [[nodiscard]] static match::Folder makeFolder(const StaticFolder<M> &folder)
    {
        return [&](auto &&... args) {
            return folder(std::forward<decltype(args)>(args)...);
        };
    }
    [[nodiscard]] static match::Folder makeFolder(auto &&pattern)
    {
        return match::makeFolder(std::forward<decltype(pattern)>(pattern));
    }

public:
    /// Initializes a StaticFolder from @p fns .
    /*implicit*/ StaticFolder(auto &&... fns)
            : Base{makeFolder(std::forward<decltype(fns)>(fns))...}
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

private:
    /// Unrolls the execution of all contained fold callbacks.
    template<std::size_t I = 0>
    LogicalResult unroll(
        Operation* op,
        ArrayRef<Attribute> operands,
        SmallVectorImpl<OpFoldResult> &result) const
    {
        if constexpr (I == N)
            return failure();
        else {
            if (succeeded(Base::operator[](I)(op, operands, result)))
                return success();
            return unroll<I + 1>(op, operands, result);
        }
    }
};

// CTAD for StaticFolder.
template<class... Patterns>
StaticFolder(Patterns &&...) -> StaticFolder<sizeof...(Patterns)>;

} // namespace mlir::ub
