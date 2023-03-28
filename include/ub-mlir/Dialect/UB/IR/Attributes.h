/// Declaration of the UB dialect attributes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/TypeUtilities.h"
#include "ub-mlir/Dialect/UB/IR/Base.h"

#include "llvm/ADT/APInt.h"

#include <algorithm>
#include <optional>

namespace mlir::ub {

/// Reference to a loaded dialect.
using DialectRef = Dialect*;

/// Concept for an attribute with an associated type.
///
/// Satisfied by TypedAttr or TypeAttr.
class TypedOrTypeAttr : public Attribute {
public:
    using Attribute::Attribute;

    // @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(TypedAttr) { return true; }
    // @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(TypeAttr) { return true; }
    /// Determines whether @p attr is a ValueOrTypeAttr.
    [[nodiscard]] static bool classof(Attribute attr)
    {
        return attr.isa<TypedAttr, TypeAttr>();
    }

    /// Builds the canonical TypedOrTypeAttr for @p attr .
    ///
    /// @pre    `attr`
    [[nodiscard]] static TypedOrTypeAttr get(TypedAttr attr) { return attr; }
    /// Builds the canonical TypedOrTypeAttr for @p attr .
    ///
    /// @pre    `attr`
    [[nodiscard]] static TypedOrTypeAttr get(TypeAttr attr) { return attr; }
    /// Builds the canonical TypedOrTypeAttr for @p type .
    ///
    /// @pre    `type`
    [[nodiscard]] static TypedOrTypeAttr get(Type type)
    {
        return TypeAttr::get(type);
    }

    /// Initializes a TypedOrTypeAttr from @p attr .
    ///
    /// @pre    `attr`
    /*implicit*/ TypedOrTypeAttr(TypedAttr attr)
            : Attribute(attr.cast<Attribute>().getImpl())
    {}
    /// Initializes a TypedOrTypeAttr from @p attr .
    ///
    /// @pre    `attr`
    /*implicit*/ TypedOrTypeAttr(TypeAttr attr)
            : Attribute(attr.cast<Attribute>().getImpl())
    {}

    /// Gets the underlying type.
    [[nodiscard]] Type getType() const
    {
        if (const auto typeAttr = dyn_cast<TypeAttr>())
            return typeAttr.getValue();

        return cast<TypedAttr>().getType();
    }
};

} // namespace mlir::ub

//===- Generated includes -------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "ub-mlir/Dialect/UB/IR/Attributes.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ub {

/// Concept for an attribute that is either @p ValueAttr or a PoisonAttr of it.
///
/// @warning @p ValueAttr must be a TypedAttr!
template<class ValueAttr>
class ValueOrPoisonAttr : public Attribute {
public:
    using Attribute::Attribute;

    // @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ValueAttr) { return true; }
    // @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(PoisonAttr poisonAttr)
    {
        return poisonAttr.isPoison()
               || poisonAttr.getSourceAttr().isa<ValueAttr>();
    }
    /// Determines whether @p attr is a ValueOrPoisonAttr.
    [[nodiscard]] static bool classof(Attribute attr)
    {
        if (attr.isa<ValueAttr>()) return true;
        if (const auto poisonAttr = attr.dyn_cast<PoisonAttr>())
            return classof(poisonAttr);
        return false;
    }

    /// Builds the fully-poisoned attribute for @p type .
    ///
    /// @pre    `type`
    [[nodiscard]] static ValueOrPoisonAttr get(Type type)
    {
        return PoisonAttr::get(type).cast<ValueOrPoisonAttr>();
    }
    /// Builds a (partially) poisoned attribute for @p sourceAttr .
    ///
    /// @pre    `sourceDialect`
    /// @pre    `sourceAttr`
    [[nodiscard]] static ValueOrPoisonAttr
    get(DialectRef sourceDialect,
        ValueOrPoisonAttr sourceAttr,
        llvm::APInt poisonMask)
    {
        assert(sourceDialect);
        assert(sourceAttr);

        // NOTE: If the union of poison masks is 0, we could return the
        //       ValueAttr directly. If we don't, the Unpoison canonicalization
        //       will get it though.
        return PoisonAttr::get(sourceDialect, sourceAttr, poisonMask);
    }
    /// Builds a (partially) poisoned attribute for @p sourceAttr .
    ///
    /// @pre    @p sourceDialect names a registered or loaded dialect.
    /// @pre    `sourceAttr`
    [[nodiscard]] static ValueOrPoisonAttr
    get(StringRef sourceDialectName,
        ValueOrPoisonAttr sourceAttr,
        llvm::APInt poisonMask)
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

    /// Initializes a ValueOrPoisonAttr from @p attr .
    ///
    /// @pre    `attr`
    /*implicit*/ ValueOrPoisonAttr(ValueAttr attr)
            : Attribute(attr.template cast<Attribute>().getImpl())
    {}

    /// Gets the underlying type.
    [[nodiscard]] Type getType() const { return cast<TypedAttr>().getType(); }
    /// Gets the underlying source attribute.
    [[nodiscard]] TypedOrTypeAttr getSourceAttr() const
    {
        if (const auto poisonAttr = dyn_cast<PoisonAttr>())
            return poisonAttr.getSourceAttr();

        return cast<TypedAttr>();
    }
    /// Gets the underlying value attribute, if any.
    [[nodiscard]] ValueAttr getValueAttr() const
    {
        return getSourceAttr().template dyn_cast<ValueAttr>();
    }

    /// Gets the poisoned element mask.
    [[nodiscard]] llvm::APInt getPoisonMask() const
    {
        if (const auto poisonAttr = dyn_cast<PoisonAttr>())
            return poisonAttr.getPoisonMask();

        return llvm::APInt(0U, 0UL);
    }
    /// Determines whether this value is (partially) poisoned.
    [[nodiscard]] bool isPoisoned() const
    {
        if (const auto poisonAttr = dyn_cast<PoisonAttr>())
            return poisonAttr.isPoisoned();

        return false;
    }
    /// Determines whether the element at @p index is poisoned.
    [[nodiscard]] bool isPoisoned(unsigned index) const
    {
        if (const auto poisonAttr = dyn_cast<PoisonAttr>())
            return poisonAttr.isPoisoned(index);

        return false;
    }
    /// Determines whether this value is fully poisoned.
    [[nodiscard]] bool isPoison() const
    {
        if (const auto poisonAttr = dyn_cast<PoisonAttr>())
            return poisonAttr.isPoison();

        return false;
    }
};

} // namespace mlir::ub
