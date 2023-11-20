//===- Attributes.td ---------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the UBX dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef UBX_ATTRIBUTES
#define UBX_ATTRIBUTES

include "ub-mlir/Dialect/UBX/IR/Base.td"

//===----------------------------------------------------------------------===//
// PoisonAttr
//===----------------------------------------------------------------------===//

def UBX_PoisonAttr : UBX_Attr<"Poison", [TypedAttrInterface]> {
    let mnemonic = "poison";
    let summary = "";
    let description = [{
        TODO
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

def UBX_PoisonedElementsAttr
        : UBX_Attr<"PoisonedElements", [TypedAttrInterface]> {
    let mnemonic = "poisoned_elements";
    let summary = "";
    let description = [{
        TODO
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

#endif
