#include "ub-mlir/Dialect/UBX/Folding/Constant.h"

#include "llvm/ADT/StringExtras.h"

#include <doctest/doctest.h>
#include <iostream>

using namespace mlir;
using namespace mlir::ubx;

class MyConst : public ConstantLike<MyConst, IntegerAttr, IntegerType> {
public:
    using ConstantLike::ConstantLike;

    [[nodiscard]] static MyConst getDense(
        ShapedType shapedTy,
        ArrayRef<llvm::APInt> values,
        ArrayRef<bool> mask = {})
    {
        assert(mask.empty() || mask.size() == values.size());

        const auto denseAttr = DenseIntElementsAttr::get(shapedTy, values);
        if (mask.empty()) return llvm::cast<MyConst>(denseAttr);

        const auto dialect =
            shapedTy.getContext()->getLoadedDialect<ubx::UBXDialect>();
        return llvm::cast<MyConst>(PoisonedElementsAttr::get(
            dialect,
            denseAttr,
            MaskAttr::get(shapedTy, mask)));
    }
};

TEST_CASE("asd")
{
    MLIRContext ctx;
    ctx.loadDialect<ubx::UBXDialect>();

    const auto i64 = IntegerType::get(&ctx, 64);
    const auto v3_i64 = VectorType::get({3}, i64);

    const auto i64_psn = MyConst::get(i64, std::nullopt);
    const auto i64_0 = MyConst::get(i64, llvm::APInt(64U, 0));
    const auto i64_1 = MyConst::get(i64, llvm::APInt(64U, 1));
    const auto v3_i64_psn = MyConst::get(v3_i64, std::nullopt);
    const auto v3_i64_012 = MyConst::getDense(
        v3_i64,
        {llvm::APInt(64U, 0), llvm::APInt(64U, 1), llvm::APInt(64U, 2)});
    const auto v3_i64_x1x = MyConst::getDense(
        v3_i64,
        {llvm::APInt(64U, 0), llvm::APInt(64U, 1), llvm::APInt(64U, 0)},
        {true, false, true});

    const auto add =
        [](std::optional<llvm::APInt> lhs,
           std::optional<llvm::APInt> rhs) -> std::optional<llvm::APInt> {
        if (!lhs || !rhs) return std::nullopt;
        return *lhs + *rhs;
    };
    const auto add_s = [](std::optional<llvm::APInt> lhs,
                          std::optional<llvm::APInt> rhs) -> llvm::APInt {
        if (!lhs) {
            if (rhs) return *rhs;
            return llvm::APInt(64U, 0);
        }
        if (!rhs) return *lhs;
        return *lhs + *rhs;
    };
    const auto print = [](std::optional<llvm::APInt> x) {
        if (!x) {
            llvm::errs() << "X, ";
            return;
        }
        llvm::errs() << x->getZExtValue() << ", ";
    };

    ubx::apply(print, i64_psn);
    llvm::errs() << "\n";
    ubx::apply(print, i64_0);
    llvm::errs() << "\n";
    ubx::apply(print, i64_1);
    llvm::errs() << "\n";
    ubx::apply(print, v3_i64_psn);
    llvm::errs() << "\n";
    ubx::apply(print, v3_i64_012);
    llvm::errs() << "\n";
    ubx::apply(print, v3_i64_x1x);
    llvm::errs() << "\n";
}
