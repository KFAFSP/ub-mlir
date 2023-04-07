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

def UB_PoisonAttr : UB_Attr<"Poison", [TypedAttrInterface]> {
    let mnemonic = "poison";
    let summary = "Poisoned value attribute";
    let description = [{
        The `poison` attribute is a compile-time proposition that the result of
        a computation invokes undefined behavior. It may also carry an
        underlying container constant value that is only partially poisoned.

        Syntax:

        ```
        poison-attr ::= (`<` (poison-spec `,`)? source-attr `>`)? `:` type
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

    // NOTE: The AttributeSelfTypeParameter is deliberately unused since that is
    //       stored within $sourceAttr.
    let parameters = (ins
        "DialectRef":$sourceDialect,
        "TypedOrTypeAttr":$sourceAttr,
        "PoisonMask":$poisonMask);

    let builders = [
        AttrBuilderWithInferredContext<(ins "Type":$type),
        [{
            return $_get(
                type.getContext(),
                nullptr,
                TypeAttr::get(type),
                PoisonMask{});
        }]>,
        AttrBuilderWithInferredContext<(ins
            "DialectRef":$sourceDialect,
            "TypedOrTypeAttr":$sourceAttr,
            CArg<"const PoisonMask &", "{}">:$poisonMask)>,
        AttrBuilderWithInferredContext<(ins
            "StringRef":$sourceDialectName,
            "TypedOrTypeAttr":$sourceAttr,
            CArg<"const PoisonMask &", "{}">:$poisonMask),
        [{
            using Base = PoisonAttr;
            return $_get(
                sourceAttr.getContext()->getOrLoadDialect(sourceDialectName),
                sourceAttr,
                poisonMask);
        }]>];
    let skipDefaultBuilders = 1;

    // NOTE: There seems to be no way to use assemblyFormat to implement the
    //       desired format (stripped value attr).
    let hasCustomAssemblyFormat = 1;

    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
    public:
        /// Determines whether the value is (partially) poisoned.
        [[nodiscard]] bool isPoisoned() const
        {
            return !getSourceDialect() || getPoisonMask().isPoisoned();
        }
        /// Determines whether the value at @p index is poisoned.
        [[nodiscard]] bool isPoisoned(unsigned index) const
        {
            return !getSourceDialect() || getPoisonMask().isPoisoned(index);
        }
        /// Determines whether the value is fully poisoned.
        [[nodiscard]] bool isPoison() const { return !getSourceDialect(); }

        //===--------------------------------------------------------------===//
        // TypedAttrInterface
        //===--------------------------------------------------------------===//

        /// Gets the underlying attribute value type.
        [[nodiscard]] Type getType() const { return getSourceAttr().getType(); }
    }];
}

//===----------------------------------------------------------------------===//
// NeverAttr
//===----------------------------------------------------------------------===//

def UB_NeverAttr : UB_Attr<"Never", [TypedAttrInterface]> {
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
        #ub.never : i16
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
