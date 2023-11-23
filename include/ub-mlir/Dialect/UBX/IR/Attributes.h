/// Declaration of the UBX dialect attributes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "ub-mlir/Dialect/UBX/IR/Base.h"
#include "ub-mlir/Dialect/UBX/Interfaces/PoisonAttrInterface.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator.h"

namespace mlir::ubx {

/// Reference to a loaded dialect.
using DialectRef = Dialect *;

/// Concept for an Attribute that stores an element-wise mask.
///
/// Satisfied by an ElementsAttr that holds `i1` elements.
class MaskAttr : public ElementsAttr {
public:
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ElementsAttr attr)
    {
        // Expect the boolean type.
        return attr.getElementType().isSignlessInteger(1);
    }
    /// Determines whether @p attr is a MaskAttr.
    ///
    /// @pre    `attr`
    [[nodiscard]] static bool classof(Attribute attr)
    {
        if (const auto elements = llvm::dyn_cast<ElementsAttr>(attr))
            return classof(elements);

        return false;
    }

    /// Obtains a MaskAttr with @p shape and @p values .
    ///
    /// @pre    @p shape is static.
    /// @pre    @p values match @p shape , or indicate a splat.
    [[nodiscard]] static MaskAttr
    get(MLIRContext *ctx, ArrayRef<int64_t> shape, ArrayRef<bool> values)
    {
        const auto i1Ty = IntegerType::get(ctx, 1);
        const auto shapedTy = RankedTensorType::get(shape, i1Ty);
        return llvm::cast<MaskAttr>(DenseElementsAttr::get(shapedTy, values));
    }

    /// Obtains a MaskAttr for a value of @p shapedTy using @p values .
    ///
    /// @pre    `shapedTy`
    /// @pre    @p values match @p shapedTy , or indicate a splat.
    [[nodiscard]] static MaskAttr
    get(ShapedType shapedTy, ArrayRef<bool> values)
    {
        assert(shapedTy);
        assert(shapedTy.hasStaticShape());

        return get(shapedTy.getContext(), shapedTy.getShape(), values);
    }

    /// Obtains a MaskAttr for @p shape and @p splatValue .
    ///
    /// @pre    @p shape is static.
    [[nodiscard]] static MaskAttr
    get(MLIRContext *ctx, ArrayRef<int64_t> shape, bool splatValue)
    {
        return get(ctx, shape, ArrayRef<bool>(splatValue));
    }

    /// Obtains a MaskAttr for a value of @p shapedTy using @p splatValue .
    ///
    /// @pre    `shapedTy`
    [[nodiscard]] static MaskAttr get(ShapedType shapedTy, bool splatValue)
    {
        return get(shapedTy, ArrayRef<bool>(splatValue));
    }

    using ElementsAttr::ElementsAttr;

    using ElementsAttr::isSplat;
    /// Determines whether all contained elements are equal to @p value .
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isSplat(bool value) const
    {
        return isSplat() && getSplatValue() == value;
    }

    /// Gets an ElementsAttrIndexer for the contained boolean values.
    ///
    /// @pre    `*this`
    [[nodiscard]] mlir::detail::ElementsAttrIndexer getIndexer() const
    {
        auto result = getValuesImpl(TypeID::get<bool>());
        assert(succeeded(result));
        return std::move(result).value();
    }

    //===------------------------------------------------------------------===//
    // ElementsAttr shadowing
    //===------------------------------------------------------------------===//
    //
    // Users should only use this concept to access boolean values, so we shadow
    // the interface of ElementsAttr to concretize all templates to bool.
    //

    using ContiguousIterableTypesT = std::tuple<>;
    using NonContiguousIterableTypesT = std::tuple<bool>;

    using iterator = ElementsAttr::iterator<bool>;
    using iterator_range = ElementsAttr::iterator_range<bool>;

    [[nodiscard]] bool getSplatValue() const
    {
        return ElementsAttr::getSplatValue<bool>();
    }

    [[nodiscard]] iterator_range getValues() const
    {
        return ElementsAttr::getValues<bool>();
    }
    [[nodiscard]] iterator value_begin() const
    {
        return ElementsAttr::value_begin<bool>();
    }
    [[nodiscard]] iterator value_end() const
    {
        return ElementsAttr::value_end<bool>();
    }
};

namespace detail {

/// Implements an iterator for the values contained in a PoisonedElementsAttr.
template<class T>
class PoisonedElementsAttrIterator : public llvm::iterator_facade_base<
                                         PoisonedElementsAttrIterator<T>,
                                         std::random_access_iterator_tag,
                                         std::optional<T>,
                                         std::ptrdiff_t,
                                         std::optional<T>,
                                         std::optional<T>> {
public:
    PoisonedElementsAttrIterator(
        mlir::detail::ElementsAttrIndexer elements,
        mlir::detail::ElementsAttrIndexer mask,
        std::size_t index)
            : m_elements(std::move(elements)),
              m_mask(std::move(mask)),
              m_index(index)
    {}

    [[nodiscard]] std::ptrdiff_t
    operator-(const PoisonedElementsAttrIterator &rhs) const
    {
        return m_index - rhs.m_index;
    }
    [[nodiscard]] bool operator==(const PoisonedElementsAttrIterator &rhs) const
    {
        return m_index == rhs.m_index;
    }
    [[nodiscard]] bool operator<(const PoisonedElementsAttrIterator &rhs) const
    {
        return m_index < rhs.m_index;
    }
    PoisonedElementsAttrIterator &operator+=(ptrdiff_t offset)
    {
        m_index += offset;
        return *this;
    }
    PoisonedElementsAttrIterator &operator-=(ptrdiff_t offset)
    {
        m_index -= offset;
        return *this;
    }

    [[nodiscard]] std::optional<T> operator*() const
    {
        if (m_mask.at<bool>(m_index)) return std::nullopt;
        return std::optional<T>(std::in_place, m_elements.at<T>(m_index));
    }

private:
    mlir::detail::ElementsAttrIndexer m_elements;
    mlir::detail::ElementsAttrIndexer m_mask;
    std::ptrdiff_t m_index;
};

/// Obtains an iterator over the @p elements poisoned by @p mask .
///
/// If @p elements is `nullptr`, @p mask must be a splat of `true` values, and
/// the iterator will only return `std::nullopt`.
///
/// @pre    `elements || mask.isSplat(true)`
/// @pre    `mask`
template<class T>
FailureOr<PoisonedElementsAttrIterator<T>>
try_value_begin(ElementsAttr elements, MaskAttr mask)
{
    if (!elements) {
        assert(mask.isSplat(true));

        // In this scenario, the default-constructed elements indexer is never
        // used due to the mask being a splat of true.
        return success(PoisonedElementsAttrIterator<T>(
            mlir::detail::ElementsAttrIndexer(),
            mask.getIndexer(),
            0));
    }

    auto elementIndexer = elements.getValuesImpl(TypeID::get<T>());
    if (succeeded(elementIndexer)) {
        return success(PoisonedElementsAttrIterator<T>(
            std::move(elementIndexer).value(),
            mask.getIndexer(),
            0));
    }

    return failure();
}

} // namespace detail

} // namespace mlir::ubx

//===- Generated includes -------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "ub-mlir/Dialect/UBX/IR/Attributes.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ubx {

/// Concept for an Attribute that is either an ElementsAttr or poison.
///
/// Satisfied by an ElementsAttr, PoisonAttr or PoisonedElementsAttr that has an
/// element type matching @p ElementType .
template<TypeConstraint ElementType = Type>
class ElementsOrPoisonAttr : public Attribute {
public:
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ElementsAttr attr)
    {
        return llvm::isa<ElementType>(attr.getElementType());
    }
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(PoisonedElementsAttr attr)
    {
        return llvm::isa<ElementType>(attr.getElementType());
    }
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(PoisonAttr attr)
    {
        if (const auto shaped = llvm::dyn_cast<ShapedType>(attr.getType()))
            return llvm::isa<ElementType>(shaped.getElementType());

        return false;
    }
    /// Determines whether @p attr is an ElementsOrPoisonAttr.
    ///
    /// @pre    `attr`
    [[nodiscard]] static bool classof(Attribute attr)
    {
        return llvm::TypeSwitch<Attribute, bool>(attr)
            .Case([](ElementsAttr attr) { return classof(attr); })
            .Case([](PoisonedElementsAttr attr) { return classof(attr); })
            .Case([](PoisonAttr attr) { return classof(attr); })
            .Default([](auto) { return false; });
    }

    using Attribute::Attribute;

    /// Initializes an ElementsOrPoisonAttr from @p attr .
    ///
    /// Only participates in overload resolution if @p ElementType is a
    /// tautological constraint.
    ///
    /// @pre    `attr`
    /*implicit*/ ElementsOrPoisonAttr(ElementsAttr attr)
        requires(std::is_same_v<ElementType, Type>)
            : Attribute(static_cast<Attribute>(attr).getImpl())
    {}

    /// Initializes an ElementsOrPoisonAttr from @p attr .
    ///
    /// Only participates in overload resolution if @p ElementType is a
    /// tautological constraint.
    ///
    /// @pre    `attr`
    /*implicit*/ ElementsOrPoisonAttr(PoisonedElementsAttr attr)
        requires(std::is_same_v<ElementType, Type>)
            : Attribute(static_cast<Attribute>(attr).getImpl())
    {}

    /// Obtains the poison attribute for @p shapedTy .
    ///
    /// @pre    `shapedTy`
    /// @pre    `llvm::isa<ElementType>(shapedTy.getElementType())`
    [[nodiscard]] static ElementsOrPoisonAttr
    get(ShapedType shapedTy, std::nullopt_t)
    {
        assert(llvm::isa<ElementType>(shapedTy.getElementType()));

        return llvm::cast<ElementsOrPoisonAttr>(PoisonAttr::get(shapedTy));
    }

    //===------------------------------------------------------------------===//
    // PoisonedElementsAttr-style interface
    //===------------------------------------------------------------------===//

    /// Determines whether this value is fully poisoned.
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isPoison() const
    {
        return llvm::TypeSwitch<Attribute, bool>(*this)
            .Case([](PoisonedElementsAttr attr) { return attr.isPoison(); })
            .Case([](PoisonAttr) { return true; })
            .Default([](auto) { return false; });
    }

    /// Determines whether this value contains any poison.
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isPoisoned() const
    {
        return llvm::TypeSwitch<Attribute, bool>(*this)
            .Case([](PoisonedElementsAttr attr) { return attr.isPoisoned(); })
            .Case([](PoisonAttr) { return true; })
            .Default([](auto) { return false; });
    }

    /// Determines whether this value is well-defined.
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isWellDefined() const { return !isPoisoned(); }

    /// Gets the underlying elements for this value.
    ///
    /// The result value may be `nullptr` if the value is entirely poisoned.
    ///
    /// @pre    `*this`
    [[nodiscard]] ElementsAttr getElements() const
    {
        return llvm::TypeSwitch<Attribute, ElementsAttr>(*this)
            .Case([](ElementsAttr attr) { return attr; })
            .Case([](PoisonedElementsAttr attr) { return attr.getElements(); })
            .Default([](auto) -> ElementsAttr { return {}; });
    }

    /// Gets the poison mask for this value.
    ///
    /// @pre    `*this`
    [[nodiscard]] MaskAttr getMask() const
    {
        return llvm::TypeSwitch<Attribute, MaskAttr>(*this)
            .Case([](ElementsAttr attr) {
                return MaskAttr::get(attr.getShapedType(), false);
            })
            .Case([](PoisonedElementsAttr attr) { return attr.getMask(); })
            .Case([](PoisonAttr attr) {
                return MaskAttr::get(
                    llvm::cast<ShapedType>(attr.getType()),
                    true);
            });
    }

    //===------------------------------------------------------------------===//
    // ElementsAttr-style interface
    //===------------------------------------------------------------------===//

    template<class T>
    using iterator = detail::PoisonedElementsAttrIterator<T>;
    template<class T>
    using iterator_range = mlir::detail::ElementsAttrRange<iterator<T>>;

    /// Obtains an iterator over the contained `std::optional<T>` values.
    ///
    /// @pre    `*this`
    template<class T>
    [[nodiscard]] FailureOr<iterator<T>> try_value_begin() const
    {
        return detail::try_value_begin<T>(getElements(), getMask());
    }

    /// Obtains an iterator over the contained `std::optional<T>` values.
    ///
    /// @pre    `*this`
    template<class T>
    [[nodiscard]] iterator<T> value_begin() const
    {
        return *try_value_begin<T>();
    }

    /// Obtains a range of the contained `std::optional<T>` values.
    ///
    /// @pre    `*this`
    template<typename T>
    [[nodiscard]] std::optional<iterator_range<T>> tryGetValues() const
    {
        if (auto begin = try_value_begin<T>()) {
            return iterator_range<T>(
                getType(),
                begin,
                std::next(begin, size()));
        }

        return std::nullopt;
    }

    /// Obtains a range of the contained `std::optional<T>` values.
    ///
    /// @pre    `*this`
    template<typename T>
    [[nodiscard]] iterator_range<T> getValues() const
    {
        return *tryGetValues<T>();
    }

    /// Determines whether all contained values are known to be equal.
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isSplat() const
    {
        return llvm::TypeSwitch<Attribute, bool>(*this)
            .Case([](ElementsAttr attr) { return attr.isSplat(); })
            .Case([](PoisonedElementsAttr attr) { return attr.isSplat(); })
            .Case([](PoisonAttr) { return true; });
    }

    /// Gets the number of elements.
    ///
    /// @pre    `*this`
    [[nodiscard]] std::size_t size() const
    {
        return getType().getNumElements();
    }

    /// Determines whether no elements are contained.
    [[nodiscard]] bool empty() const { return size() == 0; }

    /// Gets the type of the contained aggregate value.
    ///
    /// @pre    `*this`
    [[nodiscard]] ShapedType getShapedType() const
    {
        return llvm::TypeSwitch<Attribute, ShapedType>(*this)
            .Case([](ElementsAttr attr) { return attr.getType(); })
            .Case([](TypedAttr attr) {
                return llvm::cast<ShapedType>(attr.getType());
            });
    }

    /// Gets the type of the contained elements.
    ///
    /// @pre    `*this`
    [[nodiscard]] ElementType getElementType() const
    {
        return llvm::cast<ElementType>(getType().getElementType());
    }

    /// @copydoc getShapedType()
    [[nodiscard]] ShapedType getType() const { return getShapedType(); }
};

namespace detail {

template<class T>
struct maybe_optional {
    using type = std::optional<T>;
};

template<>
struct maybe_optional<void> {
    using type = void;
};

template<class T>
using maybe_optional_t = typename maybe_optional<T>::type;

} // namespace detail

/// Concept for an Attribute that is either @p ValueAttr or poison.
///
/// Satisfied by a ValueAttr or PoisonAttr that has a type matching @p Type .
///
/// @pre    `requires (ValueAttr valueAttr) { valueAttr.getType() }`
template<AttrConstraint ValueAttr, TypeConstraint Type = mlir::Type>
class ValueOrPoisonAttr : public Attribute {
    static_assert(
        requires(ValueAttr attr) { attr.getType(); },
        "ValueAttr must be a TypedAttr.");

public:
    /// The underlying value attribute value type, or void.
    using DataType = typename ValueAttr::ValueType;
    /// The value type of this attribute, or void.
    ///
    /// Wraps the DataType in an std::optional, if not void.
    using ValueType = detail::maybe_optional_t<DataType>;

    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ValueAttr attr)
    {
        return llvm::isa<Type>(attr.getType());
    }
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(PoisonAttr attr)
    {
        return llvm::isa<Type>(attr.getType());
    }
    /// Determines whether @p attr is a ValueOrPoisonAttr.
    ///
    /// @pre    `attr`
    [[nodiscard]] static bool classof(Attribute attr)
    {
        return llvm::TypeSwitch<Attribute, bool>(attr)
            .Case([](PoisonAttr attr) { return classof(attr); })
            .Case([](ValueAttr attr) { return classof(attr); })
            .Default([](auto) { return false; });
    }

    using Attribute::Attribute;

    /// Initializes a ValueOrPoisonAttr from @p attr .
    ///
    /// Only participates in overload resolution if @p Type is a tautological
    /// constraint on `attr.getType()`.
    ///
    /// @pre    `attr`
    /*implicit*/ ValueOrPoisonAttr(ValueAttr attr)
        requires(std::is_base_of_v<Type, decltype(attr.getType())>)
            : Attribute(static_cast<Attribute>(attr).getImpl())
    {}

    /// Initializes a ValueOrPoisonAttr from @p attr .
    ///
    /// Only participates in overload resolution if @p Type is a tautological
    /// constraint.
    ///
    /// @pre    `attr`
    /*implicit*/ ValueOrPoisonAttr(PoisonAttr attr)
        requires(std::is_same_v<Type, mlir::Type>)
            : Attribute(static_cast<Attribute>(attr).getImpl())
    {}

    /// Obtains the poison atttribute for @p type .
    ///
    /// @pre    `type`
    [[nodiscard]] static ValueOrPoisonAttr get(Type type, std::nullopt_t)
    {
        return llvm::cast<ValueOrPoisonAttr>(PoisonAttr::get(type));
    }
    /// Obtains the value attribute for @p type and @p value .
    ///
    /// Only participate in overload resultion if `ValueAttr::get` can be
    /// invoked on @p type and @p value .
    ///
    /// @pre    `type`
    [[nodiscard]] static ValueOrPoisonAttr get(Type type, DataType value)
        requires requires { ValueAttr::get(type, value); }
    {
        return llvm::cast<ValueOrPoisonAttr>(ValueAttr::get(type, value));
    }
    /// Obtains the ValueOrPoisonAttr for @p type and @p maybeValue .
    ///
    /// Only participate in overload resultion if `ValueAttr::get` can be
    /// invoked on @p type and the contents of @p maybeValue .
    ///
    /// @pre    `type`
    [[nodiscard]] static ValueOrPoisonAttr get(Type type, ValueType maybeValue)
        requires requires { ValueAttr::get(type, *maybeValue); }
    {
        return maybeValue ? get(type, *maybeValue) : get(type, std::nullopt);
    }

    /// Determines whether this value is poison.
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isPoison() const { return llvm::isa<PoisonAttr>(*this); }

    /// Determines whether this value contains any poison.
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isPoisoned() const { return isPoison(); }

    /// Determines whether this value is well-defined.
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isWellDefined() const { return !isPoisoned(); }

    /// Gets the contained value.
    ///
    /// Wraps the underlying attribute value as an std::optional that is absent
    /// if poisoned. Only participates in overload resolution when that type is
    /// well-defined.
    ///
    /// @pre    `*this`
    [[nodiscard]] ValueType getValue() const
        requires(!std::is_void_v<ValueType>)
    {
        if (isPoison()) return std::nullopt;
        const auto valueAttr = llvm::cast<ValueAttr>(*this);
        return std::optional<DataType>(std::in_place, valueAttr.getValue());
    }

    /// Gets the value type.
    ///
    /// @pre    `*this`
    [[nodiscard]] Type getType() const
    {
        if (const auto poison = llvm::dyn_cast<PoisonAttr>(*this))
            return llvm::cast<Type>(poison.getType());

        return llvm::cast<Type>(llvm::cast<ValueAttr>(*this).getType());
    }
};

namespace detail {

/// Implements an "iterator" that always yields the same value.
template<class T>
struct SingletonIterator : llvm::iterator_facade_base<
                               SingletonIterator<T>,
                               std::forward_iterator_tag,
                               T> {
    /*implicit*/ SingletonIterator(T value) : m_value(std::move(value)) {}

    [[nodiscard]] const T &operator*() const { return m_value; }
    SingletonIterator &operator++() { return *this; }

    [[nodiscard]] bool operator==(const SingletonIterator &) const = default;

private:
    T m_value;
};

/// Obtains an iterator over the @p splat .
///
/// @pre    `splat`
template<class ValueAttr, class Type>
    requires(!std::is_void_v<typename ValueAttr::ValueType>)
FailureOr<PoisonedElementsAttrIterator<typename ValueAttr::ValueType>>
try_value_begin(ValueOrPoisonAttr<ValueAttr, Type> splat)
{
    static constexpr std::array verum{true};
    static constexpr std::array falsum{false};

    assert(splat);

    using T = typename ValueAttr::ValueType;

    if (splat.isPoison()) {
        return success(PoisonedElementsAttrIterator<T>(
            mlir::detail::ElementsAttrIndexer(),
            mlir::detail::ElementsAttrIndexer::contiguous(true, verum.data()),
            0));
    }

    return success(PoisonedElementsAttrIterator<T>(
        mlir::detail::ElementsAttrIndexer::nonContiguous(
            true,
            SingletonIterator<T>(*splat.getValue())),
        mlir::detail::ElementsAttrIndexer::contiguous(true, falsum.data()),
        0));
}

/// Marker for the ValueOrPoisonLikeAttr concept.
struct ValueOrPoisonLikeAttrBase {};

} // namespace detail

/// Concept for an Attribute that is either @p ValueAttr , a container for it,
/// or poison.
///
/// Satisfied by a ValueOrPoisonAttr or ElementsOrPoisonAttr.
///
/// @pre    `requires (ValueAttr valueAttr) { valueAttr.getType() }`
template<AttrConstraint ValueAttr, TypeConstraint ElementType = mlir::Type>
class ValueOrPoisonLikeAttr : public Attribute,
                              public detail::ValueOrPoisonLikeAttrBase {
    static_assert(
        requires(ValueAttr attr) { attr.getType(); },
        "ValueAttr must be a TypedAttr.");

public:
    /// The compatible ValueOrPoisonAttr.
    using ElementAttr = ValueOrPoisonAttr<ValueAttr, Type>;
    /// The compatible ElementsOrPoisonAttr.
    using ElementsAttr = ElementsOrPoisonAttr<Type>;

    /// The underlying value attribute value type.
    using DataType = typename ElementAttr::DataType;
    /// The value type of the compatible ElementAttr.
    using FoldType = typename ElementAttr::ValueType;

    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ElementAttr) { return true; }
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ElementsAttr) { return true; }
    /// Determines whether @p attr is a ValueOrPoisonLikeAttr.
    ///
    /// @pre    `attr`
    [[nodiscard]] static bool classof(Attribute attr)
    {
        return llvm::isa<ElementAttr, ElementsAttr>(attr);
    }

    using Attribute::Attribute;

    /// Initializes a ValueOrPoisonLikeAttr from @p attr .
    ///
    /// @pre    `attr`
    /*implicit*/ ValueOrPoisonLikeAttr(ElementAttr attr)
            : Attribute(static_cast<Attribute>(attr).getImpl())
    {}

    /// Initializes a ValueOrPoisonLikeAttr from @p attr .
    ///
    /// @pre    `attr`
    /*implicit*/ ValueOrPoisonLikeAttr(ElementsAttr attr)
            : Attribute(static_cast<Attribute>(attr).getImpl())
    {}

    /// Obtains the poison attribute for @p type .
    ///
    /// @pre    `type`
    [[nodiscard]] static ValueOrPoisonLikeAttr
    get(ElementType type, std::nullopt_t)
    {
        return ElementAttr::get(type, std::nullopt);
    }
    /// Obtains the value attribute for @p type and @p value .
    ///
    /// @pre    `type`
    [[nodiscard]] static ValueOrPoisonLikeAttr
    get(ElementType type, DataType value)
    {
        return ElementAttr::get(type, value);
    }
    /// Obtains the value attribute for @p type and @p maybeValue .
    ///
    /// @pre    `type`
    [[nodiscard]] static ValueOrPoisonLikeAttr
    get(ElementType type, FoldType maybeValue)
    {
        return ElementAttr::get(type, maybeValue);
    }
    /// Obtains the poison attribute for @p shapedTy .
    ///
    /// @pre    `shapedTy`
    /// @pre    `llvm::isa<ElementType>(shapedTy.getElementType())`
    [[nodiscard]] static ValueOrPoisonLikeAttr
    get(ShapedType shapedTy, std::nullopt_t)
    {
        return ElementsAttr::get(shapedTy, std::nullopt);
    }

    /// Determines whether this value is fully poisoned.
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isPoison() const
    {
        return llvm::TypeSwitch<Attribute, bool>(*this)
            .Case([](ElementAttr attr) { return attr.isPoison(); })
            .Case([](ElementsAttr attr) { return attr.isPoison(); });
    }

    /// Determines whether this value contains any poison.
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isPoisoned() const
    {
        return llvm::TypeSwitch<Attribute, bool>(*this)
            .Case([](ElementAttr attr) { return attr.isPoisoned(); })
            .Case([](ElementsAttr attr) { return attr.isPoisoned(); });
    }

    /// Determines whether this value is well-defined.
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isWellDefined() const { return !isPoisoned(); }

    /// Gets the attribute type.
    ///
    /// @pre    `*this`
    [[nodiscard]] Type getType() const
    {
        return llvm::TypeSwitch<Attribute, Type>(*this)
            .Case([](ElementAttr attr) { return attr.getType(); })
            .Case([](ElementsAttr attr) { return attr.getType(); });
    }

    /// Gets the underlying element type.
    ///
    /// @pre    `*this`
    [[nodiscard]] ElementType getElementType() const
    {
        return llvm::TypeSwitch<Attribute, ElementType>(*this)
            .Case([](ElementAttr attr) { return attr.getType(); })
            .Case([](ElementsAttr attr) { return attr.getElementType(); });
    }

    //===------------------------------------------------------------------===//
    // ElementsAttr-style interface
    //===------------------------------------------------------------------===//

    using iterator = detail::PoisonedElementsAttrIterator<DataType>;
    using iterator_range = mlir::detail::ElementsAttrRange<iterator>;

    /// Obtains an iterator over the contained ElementType values.
    ///
    /// @pre    `*this`
    [[nodiscard]] FailureOr<iterator> try_value_begin() const
    {
        if (const auto elements = llvm::dyn_cast<ElementsAttr>(*this))
            return elements.template try_value_begin<DataType>();

        return detail::try_value_begin(llvm::cast<ElementAttr>(*this));
    }

    /// Obtains an iterator over the contained ElementType values.
    ///
    /// @pre    `*this`
    [[nodiscard]] iterator value_begin() const { return *try_value_begin(); }

    /// Obtains a range of the contained ElementType values.
    ///
    /// @pre    `*this`
    [[nodiscard]] std::optional<iterator_range> tryGetValues() const
    {
        if (auto begin = try_value_begin())
            return iterator_range(getType(), begin, std::next(begin, size()));

        return std::nullopt;
    }

    /// Obtains a range of the contained ElementType values.
    ///
    /// @pre    `*this`
    [[nodiscard]] iterator_range getValues() const { return *tryGetValues(); }

    /// Determines whether all contained values are known to be equal.
    ///
    /// @pre    `*this`
    [[nodiscard]] bool isSplat() const
    {
        return llvm::TypeSwitch<Attribute, bool>(*this)
            .Case([](ElementAttr) { return true; })
            .Case([](ElementsAttr attr) { return attr.isSplat(); });
    }

    /// Gets the number of elements.
    ///
    /// @pre    `*this`
    [[nodiscard]] std::size_t size() const
    {
        return llvm::TypeSwitch<Attribute, std::size_t>(*this)
            .Case([](PoisonAttr attr) -> std::size_t {
                if (auto shapedTy = llvm::dyn_cast<ShapedType>(attr.getType()))
                    return shapedTy.getNumElements();
                return 1;
            })
            .Case([](ValueAttr) -> std::size_t { return 1; })
            .Case([](ElementsAttr attr) { return attr.size(); });
    }

    /// Determines whether no elements are contained.
    [[nodiscard]] bool empty() const { return size() == 0; }
};

} // namespace mlir::ubx
