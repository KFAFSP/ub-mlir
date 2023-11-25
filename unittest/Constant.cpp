#include "ub-mlir/Dialect/UBX/Folding/Constant.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <doctest/doctest.h>
#include <iostream>
#include <string>

using namespace mlir;
using namespace mlir::ubx;

namespace {

class MyConst : public ConstantLike<MyConst, IntegerAttr, IntegerType> {
public:
    using ConstantLike::ConstantLike;

    using ConstantLike::get;
    [[nodiscard]] static MyConst
    get(ShapedType shapedTy,
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

    std::string toStringZExt() const
    {
        std::string result;
        llvm::raw_string_ostream stream(result);
        const auto isShaped = llvm::isa<ShapedType>(getType());
        if (isShaped) stream << "[";
        llvm::interleaveComma(
            getValues(),
            stream,
            [&](std::optional<llvm::APInt> value) {
                if (value)
                    stream << value->getZExtValue();
                else
                    stream << "psn";
            });
        if (isShaped) stream << "]";
        return result;
    }
};

} // namespace

TEST_CASE("ConstantLike")
{
    MLIRContext ctx;
    ctx.loadDialect<ubx::UBXDialect>();

    const auto i64 = IntegerType::get(&ctx, 64);
    const auto v3_i64 = VectorType::get({3}, i64);

    const auto i64_psn = MyConst::get(i64, std::nullopt);
    const auto i64_0 = MyConst::get(i64, llvm::APInt(64U, 0));
    const auto i64_1 = MyConst::get(i64, llvm::APInt(64U, 1));
    const auto v3_i64_psn = MyConst::getSplat(v3_i64, std::nullopt);
    const auto v3_i64_012 = MyConst::get(
        v3_i64,
        {llvm::APInt(64U, 0), llvm::APInt(64U, 1), llvm::APInt(64U, 2)});
    const auto v3_i64_x1x = MyConst::get(
        v3_i64,
        {llvm::APInt(64U, 0), llvm::APInt(64U, 1), llvm::APInt(64U, 0)},
        {true, false, true});

    CHECK_EQ(i64_psn.toStringZExt(), "psn");
    CHECK_EQ(i64_0.toStringZExt(), "0");
    CHECK_EQ(i64_1.toStringZExt(), "1");
    CHECK_EQ(v3_i64_psn.toStringZExt(), "[psn, psn, psn]");
    CHECK_EQ(v3_i64_012.toStringZExt(), "[0, 1, 2]");
    CHECK_EQ(v3_i64_x1x.toStringZExt(), "[psn, 1, psn]");

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

    auto add_psn_1 = i64_psn.map({}, add, i64_1);
    auto add_s_psn_1 = i64_psn.map({}, add_s, i64_1);
    auto add_1_1 = i64_1.map({}, add, i64_1);

    CHECK_EQ(add_psn_1.toStringZExt(), "psn");
    CHECK_EQ(add_s_psn_1.toStringZExt(), "1");
    CHECK_EQ(add_1_1.toStringZExt(), "2");

    auto add_v3_psn_psn = v3_i64_psn.map({}, add, v3_i64_psn);
    auto add_s_v3_psn_psn = v3_i64_psn.map({}, add_s, v3_i64_psn);

    CHECK_EQ(add_v3_psn_psn.toStringZExt(), "[psn, psn, psn]");
    CHECK_EQ(add_s_v3_psn_psn.toStringZExt(), "[0, 0, 0]");

    auto add_v3_psn_012 = v3_i64_psn.map({}, add, v3_i64_012);
    auto add_s_v3_psn_012 = v3_i64_psn.map({}, add_s, v3_i64_012);

    CHECK_EQ(add_v3_psn_012.toStringZExt(), "[psn, psn, psn]");
    CHECK_EQ(add_s_v3_psn_012.toStringZExt(), "[0, 1, 2]");

    auto add_v3_x1x_012 = v3_i64_x1x.map({}, add, v3_i64_012);
    auto add_s_v3_x1x_012 = v3_i64_x1x.map({}, add_s, v3_i64_012);

    CHECK_EQ(add_v3_x1x_012.toStringZExt(), "[psn, 2, psn]");
    CHECK_EQ(add_s_v3_x1x_012.toStringZExt(), "[0, 2, 2]");

    auto add_v3_012_012 = v3_i64_012.map({}, add, v3_i64_012);

    CHECK_EQ(add_v3_012_012.toStringZExt(), "[0, 2, 4]");
}
