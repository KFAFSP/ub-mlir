//===- Attributes.td ---------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the UBX dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef UBX_ATTRIBUTES
#define UBX_ATTRIBUTES

include "ub-mlir/Dialect/UBX/IR/Base.td"
include "ub-mlir/Dialect/UBX/Interfaces/PoisonAttrInterface.td"

//===----------------------------------------------------------------------===//
// PoisonAttr
//===----------------------------------------------------------------------===//

def UBX_PoisonAttr : UBX_Attr<"Poison", [UBX_PoisonAttrInterface]> {
    let mnemonic = "poison";
    let summary = "Represents a poisoned value";
    let description = [{
        The `poison` attribute represents a value that is the result of
        undefined behavior. More specifically, there is no way to observe a
        poison value without observing the result of undefined behavior.

        Syntax:

        ```
        poison-attr ::= `poison` `:` type
        ```

        Example:

        ```
        #val = #ubx.poison : i32
        ```
    }];

    let parameters = (ins AttributeSelfTypeParameter<"">:$type);

    let builders = [
        AttrBuilderWithInferredContext<(ins "Type":$type),
        [{
            return $_get(type.getContext(), type);
        }]>];

    let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// PoisonedElementsAttr
//===----------------------------------------------------------------------===//

def UBX_PoisonedElementsAttr : UBX_Attr<
        "PoisonedElements",
        [DeclareAttrInterfaceMethods<UBX_PoisonAttrInterface, ["freeze"]>]> {
    let mnemonic = "poisoned_elements";
    let summary = "Represents a poisonable `ShapedType` aggregate value";
    let description = [{
        The `poisoned_elements` attribute represents a value of a `ShapedType`
        aggregate that may contain elements that are the result of undefined
        behavior. In contrast to the `poison` attribute, zero or more element
        values may be poisoned, leading to potentially well-defined observation
        of the contained value.

        The `$sourceDialect` namespace is required to refer materialization of
        the contained `$elements` back to the originator of the attribute. Using
        the same `materializeConstant` callback, the `poison` op also allows the
        source dialect to change the representation of the poisoned value during
        canonicalization.

        The type of the `$mask` attribute is implicitly given by the shape of
        the attribute type and the `i1` boolean element type. Both the
        `$elements` and `$mask` attributes are therefore parsed without type.

        Syntax:

        ```
        poisoned-elements-attr ::= `poisoned_elements` `<`
                                   $sourceDialect `(` $elements `)`
                                   `[` $mask `]` `>` `:` $type
        ```

        Example:

        ```
        #partial = #ubx.poisoned_elements<arith(dense<1>)[dense<[false, true, false]>]> : tensor<3xi64>
        ```
    }];

    // NOTE: The AttributeSelfTypeParameter is deliberately unused since that is
    //       stored within $elements.
    let parameters = (ins
        "DialectRef":$sourceDialect,
        "ElementsAttr":$elements,
        "MaskAttr":$mask);

    // NOTE: There seems to be no way to use assemblyFormat to implement the
    //       desired format (stripped elements attr).
    let hasCustomAssemblyFormat = 1;

    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
        /// Obtains the canonical Attribute for @p elements with @p mask .
        [[nodiscard]] static Attribute get(
            DialectRef sourceDialect,
            ElementsAttr elements,
            MaskAttr mask);

        /// Determines whether this value is fully poisoned.
        [[nodiscard]] bool isPoison() const
        {
            // TODO: Ensure splats are always detected?
            return getMask().isSplat(true);
        }

        //===--------------------------------------------------------------===//
        // PoisonAttrInterface
        //===--------------------------------------------------------------===//

        /// Determines whether this value contains any poison.
        [[nodiscard]] bool isPoisoned() const
        {
            // TODO: Ensure splats are always detected?
            return !getMask().isSplat(false);
        }

        //===--------------------------------------------------------------===//
        // ElementsAttr-style interface
        //===--------------------------------------------------------------===//

        template<class T>
        using iterator = detail::PoisonedElementsAttrIterator<T>;
        template<class T>
        using iterator_range = mlir::detail::ElementsAttrRange<iterator<T>>;

        /// Obtains an iterator over the contained `std::optional<T>` values.
        template<class T>
        FailureOr<iterator<T>> try_value_begin() const
        {
            return detail::try_value_begin<T>(getElements(), getMask());
        }

        /// Obtains an iterator over the contained `std::optional<T>` values.
        template<class T>
        [[nodiscard]] iterator<T> value_begin() const
        {
            return *try_value_begin<T>();
        }

        /// Obtains a range of the contained `std::optional<T>` values.
        template<typename T>
        [[nodiscard]] std::optional<iterator_range<T>> tryGetValues() const {
            if (auto begin = try_value_begin<T>()) {
                return iterator_range<T>(
                    getType(),
                    begin,
                    std::next(begin, size()));
            }

            return std::nullopt;
        }

        /// Obtains a range of the contained `std::optional<T>` values.
        template<typename T>
        [[nodiscard]] iterator_range<T> getValues() const
        {
            return *tryGetValues<T>();
        }

        /// Determines whether all contained values are known to be equal.
        [[nodiscard]] bool isSplat() const
        {
            return getMask().isSplat(true) || getElements().isSplat();
        }

        /// Gets the number of contained values.
        [[nodiscard]] std::size_t size() const { return getElements().size(); }

        /// Gets the type of the contained elements.
        [[nodiscard]] Type getElementType() const
        {
            return getElements().getElementType();
        }

        //===--------------------------------------------------------------===//
        // TypedAttr
        //===--------------------------------------------------------------===//

        /// Gets the type of the contained aggregate value.
        [[nodiscard]] ShapedType getType() const
        {
            return getElements().getShapedType();
        }
    }];
}

//===----------------------------------------------------------------------===//
// NeverAttr
//===----------------------------------------------------------------------===//

def UBX_NeverAttr : UBX_Attr<"Never", [TypedAttrInterface]> {
    let mnemonic = "never";
    let summary = "Unreachable value attribute";
    let description = [{
        The `never` attribute represents a compile-time proposition that a value
        will never become available, i.e., is the result of an unreachable
        operation.

        Syntax:

        ```
        never-attr ::= `:` type
        ```

        Example:

        ```
        #ubx.never : i16
        ```
    }];

    let parameters = (ins AttributeSelfTypeParameter<"">:$type);

    let builders = [
        AttrBuilder<(ins),
        [{
            return $_get($_ctxt, NeverType::get($_ctxt));
        }]>,
        AttrBuilderWithInferredContext<(ins "Type":$type),
        [{
            return $_get(type.getContext(), type);
        }]>];

    let assemblyFormat = "";
}

#endif
