/// Implements the ConstantLike folding concept template.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "ub-mlir/Dialect/UBX/Folding/Apply.h"
#include "ub-mlir/Dialect/UBX/IR/Attributes.h"

#include <type_traits>
#include <utility>

namespace mlir::ubx {

/// Base class for deriving constant-foldable attribute concepts.
template<
    class Derived,
    ext::AttrConstraint ValueAttr,
    ext::TypeConstraint ElementType = mlir::Type>
class ConstantLike : public ValueOrPoisonLikeAttr<ValueAttr, ElementType> {
public:
    using Base = ValueOrPoisonLikeAttr<ValueAttr, ElementType>;
    using Base::Base;

    /// The underlying value attribute value type.
    using DataType = typename Base::DataType;
    /// The value type of the compatible ElementAttr.
    using FoldType = typename Base::FoldType;

    /// Obtains the poison attribute for @p type .
    ///
    /// @pre    `type`
    [[nodiscard]] static Derived get(ElementType type, std::nullopt_t)
    {
        return llvm::cast<Derived>(Base::get(type, std::nullopt));
    }
    /// Obtains the value attribute for @p type and @p value .
    ///
    /// @pre    `type`
    [[nodiscard]] static Derived get(ElementType type, DataType value)
    {
        return llvm::cast<Derived>(Base::get(type, value));
    }
    /// Obtains the value attribute for @p type and @p maybeValue .
    ///
    /// @pre    `type`
    [[nodiscard]] static Derived get(ElementType type, FoldType maybeValue)
    {
        return llvm::cast<Derived>(Base::get(type, maybeValue));
    }
    // Derived must override this method!
    [[nodiscard]] static Derived
    get(ShapedType shapedTy,
        ArrayRef<DataType> values,
        ArrayRef<bool> mask = {});
    /// Obtains the poison attribute for @p shapedTy .
    ///
    /// @pre    `shapedTy`
    /// @pre    `llvm::isa<ElementType>(shapedTy.getElementType())`
    [[nodiscard]] static Derived getSplat(ShapedType shapedTy, std::nullopt_t)
    {
        return llvm::cast<Derived>(Base::getSplat(shapedTy, std::nullopt));
    }
    /// Obtains the constant attribute for @p shapedTy and @p splatValue .
    ///
    /// @pre    `shapedTy`
    /// @pre    `llvm::isa<ElementType>(shapedTy.getElementType())`
    [[nodiscard]] static Derived
    getSplat(ShapedType shapedTy, FoldType splatValue)
    {
        if (splatValue)
            return Derived::get(shapedTy, ArrayRef<DataType>(*splatValue), {});
        return Derived::getSplat(shapedTy, std::nullopt);
    }

    /// Applies @p fn to this and @p args .
    ///
    /// @pre    @p operands must be derived from ValueOrPoisonLikeAttr.
    /// @pre    @p fn must be invocable on the fold types of the operands.
    /// @pre    @p fn must return void.
    void apply(auto fn, auto... args) const
        requires (
            std::is_void_v<decltype(ubx::apply(std::move(fn), *this, args...))>)
    {
        ubx::apply(std::move(fn), *this, args...);
    }

    /// Applies @p fn to this and @p args , producing @p elementTy .
    ///
    /// @pre    @p operands must be derived from ValueOrPoisonLikeAttr.
    /// @pre    @p fn must be invocable on the fold types of the operands.
    /// @pre    @p fn must not return void.
    /// @pre    The shapes of all operands must match.
    [[nodiscard]] Derived
    map(ElementType elementTy, auto fn, auto... args) const
        requires (!std::is_void_v<
                  decltype(ubx::apply(std::move(fn), *this, args...))>)
    {
        if (const auto shapedTy = llvm::dyn_cast<ShapedType>(this->getType())) {
            elementTy =
                elementTy ? elementTy
                          : llvm::cast<ElementType>(shapedTy.getElementType());
            const auto resultTy = shapedTy.cloneWith(std::nullopt, elementTy);
            return _get(resultTy, ubx::apply(std::move(fn), *this, args...));
        }

        if (!elementTy) elementTy = llvm::cast<ElementType>(this->getType());
        return _get(elementTy, ubx::apply(std::move(fn), *this, args...));
    }

private:
    [[nodiscard]] static Derived
    _get(ShapedType shapedTy, SmallVector<DataType> values)
    {
        return Derived::get(shapedTy, values, {});
    }
    [[nodiscard]] static Derived _get(
        ShapedType shapedTy,
        std::pair<SmallVector<DataType>, SmallVector<bool>> values)
    {
        return Derived::get(shapedTy, values.first, values.second);
    }

    [[nodiscard]] static Derived
    _get(ElementType type, SmallVector<DataType> values)
    {
        assert(values.size() == 1);
        return Derived::get(type, values.front());
    }
    [[nodiscard]] static Derived _get(
        ElementType type,
        std::pair<SmallVector<DataType>, SmallVector<bool>> values)
    {
        assert(values.first.size() == 1);
        if (values.second.front()) return Derived::get(type, std::nullopt);
        return Derived::get(type, values.first.front());
    }
};

} // namespace mlir::ubx
