/// Declaration of the UB dialect attributes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "ub-mlir/Dialect/UB/Analysis/PoisonMask.h"
#include "ub-mlir/Dialect/UB/IR/Base.h"

#include <concepts>

namespace mlir::ub {

/// Reference to a loaded dialect.
using DialectRef = Dialect*;

/// Concept for an attribute with an associated type.
///
/// Satisfied by TypedAttr or TypeAttr.
class TypedOrTypeAttr : public Attribute {
public:
    // @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(TypedAttr) { return true; }
    // @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(TypeAttr) { return true; }
    /// Determines whether @p attr is a ValueOrTypeAttr.
    ///
    /// @pre    `attr`
    [[nodiscard]] static bool classof(Attribute attr)
    {
        return llvm::isa<TypedAttr, TypeAttr>(attr);
    }

    /// Builds the canonical TypedOrTypeAttr for @p attr .
    ///
    /// @pre    `attr`
    [[nodiscard]] static TypedOrTypeAttr get(TypedOrTypeAttr attr)
    {
        return attr;
    }
    /// Builds the canonical TypedOrTypeAttr for @p type .
    ///
    /// @pre    `type`
    [[nodiscard]] static TypedOrTypeAttr get(Type type)
    {
        return TypeAttr::get(type);
    }

    using Attribute::Attribute;

    /// Initializes a TypedOrTypeAttr from @p attr .
    ///
    /// @pre    `attr`
    /*implicit*/ TypedOrTypeAttr(TypedAttr attr)
            : Attribute(llvm::cast<Attribute>(attr).getImpl())
    {}
    /// Initializes a TypedOrTypeAttr from @p attr .
    ///
    /// @pre    `attr`
    /*implicit*/ TypedOrTypeAttr(TypeAttr attr)
            : Attribute(llvm::cast<Attribute>(attr).getImpl())
    {}

    /// Gets the underlying type.
    [[nodiscard]] Type getType() const
    {
        if (const auto typeAttr = llvm::dyn_cast<TypeAttr>(*this))
            return typeAttr.getValue();

        return llvm::cast<TypedAttr>(*this).getType();
    }
};

} // namespace mlir::ub

//===- Generated includes -------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "ub-mlir/Dialect/UB/IR/Attributes.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ub {

//===----------------------------------------------------------------------===//
// ValueOrPoisonAttr
//===----------------------------------------------------------------------===//

/// Concept for an attribute that is either @p ValueAttr or a PoisonAttr of it.
///
/// @pre    @p ValueAttr is a TypedAttr
template<std::derived_from<Attribute> ValueAttr>
class ValueOrPoisonAttr : public Attribute {
public:
    using ValueType = decltype(std::declval<ValueAttr>().getType());

    // @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ValueAttr) { return true; }
    // @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(PoisonAttr poisonAttr)
    {
        return poisonAttr.isPoison()
               || llvm::isa<ValueAttr>(poisonAttr.getSourceAttr());
    }
    /// Determines whether @p attr is a ValueOrPoisonAttr.
    ///
    /// @pre    `attr`
    [[nodiscard]] static bool classof(Attribute attr)
    {
        if (llvm::isa<ValueAttr>(attr)) return true;
        if (const auto poisonAttr = llvm::dyn_cast<PoisonAttr>(attr))
            return classof(poisonAttr);
        return false;
    }

    /// Builds the fully-poisoned attribute for @p type .
    ///
    /// @pre    `type`
    [[nodiscard]] static ValueOrPoisonAttr get(ValueType type)
    {
        return llvm::cast<ValueOrPoisonAttr>(PoisonAttr::get(type));
    }
    /// Builds a (partially) poisoned attribute for @p sourceAttr .
    ///
    /// @pre    `sourceDialect`
    /// @pre    `sourceAttr`
    [[nodiscard]] static ValueOrPoisonAttr
    get(DialectRef sourceDialect,
        ValueOrPoisonAttr sourceAttr,
        const PoisonMask &poisonMask)
    {
        assert(sourceDialect);
        assert(sourceAttr);

        // NOTE: If the union of poison masks is 0, we could return the
        //       ValueAttr directly. If we don't, the Unpoison canonicalization
        //       will get it though.
        return llvm::cast<ValueOrPoisonAttr>(PoisonAttr::get(
            sourceDialect,
            llvm::cast<TypedOrTypeAttr>(sourceAttr),
            poisonMask));
    }
    /// Builds a (partially) poisoned attribute for @p sourceAttr .
    ///
    /// @pre    @p sourceDialect names a registered or loaded dialect.
    /// @pre    `sourceAttr`
    [[nodiscard]] static ValueOrPoisonAttr
    get(StringRef sourceDialectName,
        ValueOrPoisonAttr sourceAttr,
        const PoisonMask &poisonMask)
    {
        assert(sourceAttr);

        return get(
            sourceAttr.getContext()->getOrLoadDialect(sourceDialectName),
            sourceAttr,
            poisonMask);
    }
    /// Builds an unpoisoned attribute for @p valueAttr .
    [[nodiscard]] static ValueOrPoisonAttr get(ValueAttr valueAttr)
    {
        return valueAttr;
    }

    using Attribute::Attribute;

    /// Initializes a ValueOrPoisonAttr from @p attr .
    ///
    /// @pre    `attr`
    /*implicit*/ ValueOrPoisonAttr(ValueAttr attr)
            : Attribute(llvm::cast<Attribute>(attr).getImpl())
    {}

    /// Gets the underlying type.
    [[nodiscard]] ValueType getType() const
    {
        return llvm::cast<ValueType>(llvm::cast<TypedAttr>(*this).getType());
    }
    /// Gets the underlying source attribute.
    [[nodiscard]] TypedOrTypeAttr getSourceAttr() const
    {
        if (const auto poisonAttr = llvm::dyn_cast<PoisonAttr>(*this))
            return poisonAttr.getSourceAttr();

        return llvm::cast<TypedAttr>(*this);
    }
    /// Gets the underlying value attribute, if any.
    [[nodiscard]] ValueAttr getValueAttr() const
    {
        return llvm::dyn_cast<ValueAttr>(getSourceAttr());
    }

    /// Gets the poisoned element mask.
    [[nodiscard]] PoisonMask getPoisonMask() const
    {
        if (const auto poisonAttr = llvm::dyn_cast<PoisonAttr>(*this))
            return poisonAttr.getPoisonMask();

        return {};
    }
    /// Determines whether this value is (partially) poisoned.
    [[nodiscard]] bool isPoisoned() const
    {
        if (const auto poisonAttr = llvm::dyn_cast<PoisonAttr>(*this))
            return poisonAttr.isPoisoned();

        return false;
    }
    /// Determines whether the element at @p index is poisoned.
    [[nodiscard]] bool isPoisoned(unsigned index) const
    {
        if (const auto poisonAttr = llvm::dyn_cast<PoisonAttr>(*this))
            return poisonAttr.isPoisoned(index);

        return false;
    }
    /// Determines whether this value is fully poisoned.
    [[nodiscard]] bool isPoison() const
    {
        if (const auto poisonAttr = llvm::dyn_cast<PoisonAttr>(*this))
            return poisonAttr.isPoison();

        return false;
    }
};

} // namespace mlir::ub
