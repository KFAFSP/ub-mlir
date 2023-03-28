//===- Attributes.td ---------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the UB dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef UB_ATTRIBUTES
#define UB_ATTRIBUTES

include "ub-mlir/Dialect/UB/IR/Base.td"

//===----------------------------------------------------------------------===//
// PoisonAttr
//===----------------------------------------------------------------------===//

// #ub.poison<"DEADBEEF", arith(dense<3>)> : tensor<3xi64>
// #ub.poison : tensor<3xi64>

def UB_PoisonAttr : UB_Attr<"Poison", [TypedAttrInterface]> {
    let mnemonic = "poison";
    let summary = "Poisoned value attribute";
    let description = [{
        The `poison` attribute represents a compile-time constant value that is
        the result of undefined behavior. It can also represent a constant
        container value which is only partially poisoned.

        Syntax:

        ```
        poison-attr ::= (`<` (poison-spec `,`)? source-attr `>` `:`)? type
        poison-spec ::= `"` [0-9a-fA-F]+ `"`
        source-attr ::= dialect-name `(` attr-value `)`
        ```

        Example:

        ```
        // Fully poisoned.
        #ub.poison : i16

        // Partially poisoned.
        #ub.poison<"03", arith(dense<[0,1,2]>)> : tensor<3xi16>

        // Not poisoned. Will be materialized by the source dialect.
        #ub.poison<arith(0)> : i16
        #ub.poison<arith(dense<[0,1,2]>)> : tensor<3xi16>
        ```
    }];

    let parameters = (ins
        "DialectRef":$sourceDialect,
        "Attribute":$sourceAttr,
        "llvm::APInt":$poisonMask);

    let builders = [
        AttrBuilderWithInferredContext<(ins "Type":$type),
        [{
            return $_get(
                type.getContext(),
                nullptr,
                TypeAttr::get(type),
                llvm::APInt(0U, 0UL));
        }]>,
        AttrBuilderWithInferredContext<(ins
            "DialectRef":$sourceDialect,
            "Attribute":$sourceAttr,
            "llvm::APInt":$poisonMask)>,
        AttrBuilderWithInferredContext<(ins
            "StringRef":$sourceDialectName,
            "Attribute":$sourceAttr,
            "llvm::APInt":$poisonMask),
        [{
            using Base = PoisonAttr;
            return $_get(
                sourceAttr.getContext()->getOrLoadDialect(sourceDialectName),
                sourceAttr,
                poisonMask);
        }]>];
    let skipDefaultBuilders = 1;

    let hasCustomAssemblyFormat = 1;

    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
    public:
        /// Determines whether the value is (partially) poisoned.
        [[nodiscard]] bool isPoisoned() const
        {
            return !getSourceDialect() || !getPoisonMask().isZero();
        }
        /// Determines whether the value at @p index is poisoned.
        [[nodiscard]] bool isPoisoned(unsigned index) const
        {
            if (!getSourceDialect()) return true;
            if (index >= getPoisonMask().getBitWidth()) return false;
            return getPoisonMask().tcExtractBit(
                getPoisonMask().getRawData(),
                index);
        }
        /// Determines whether the value is fully poisoned.
        [[nodiscard]] bool isPoison() const
        {
            return !getSourceDialect();
        }

        //===--------------------------------------------------------------===//
        // TypedAttrInterface
        //===--------------------------------------------------------------===//

        /// Gets the underlying attribute value type.
        [[nodiscard]] Type getType() const
        {
            if (const auto typeAttr = getSourceAttr().dyn_cast<TypeAttr>())
                return typeAttr.getValue();

            return getSourceAttr().cast<TypedAttr>().getType();
        }
    }];
}

#endif
