#include "ub-mlir/Extensions/FoldingContext.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "ub-mlir/Dialect/UBX/Folding/Constant.h"
#include "ub-mlir/Dialect/UBX/IR/UBX.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <doctest/doctest.h>
#include <iostream>
#include <string>

using namespace mlir;
using namespace mlir::ext;
using namespace mlir::ubx;

namespace {

class IntConst : public ConstantLike<IntConst, IntegerAttr, IntegerType> {
public:
    using ConstantLike::ConstantLike;

    using ConstantLike::get;
    [[nodiscard]] static IntConst
    get(ShapedType shapedTy,
        ArrayRef<llvm::APInt> values,
        ArrayRef<bool> mask = {})
    {
        assert(mask.empty() || mask.size() == values.size());

        const auto denseAttr = DenseIntElementsAttr::get(shapedTy, values);
        if (mask.empty()) return llvm::cast<IntConst>(denseAttr);

        const auto dialect =
            shapedTy.getContext()->getLoadedDialect<ubx::UBXDialect>();
        return llvm::cast<IntConst>(PoisonedElementsAttr::get(
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

class Zeros : public IntConst {
public:
    using IntConst::IntConst;

    static bool classof(Attribute attr)
    {
        const auto cst = llvm::dyn_cast<IntConst>(attr);
        if (!cst) return false;
        const auto splat = cst.tryGetSplatValue();
        if (!splat) return false;
        return splat->isZero();
    }
};

class Ones : public IntConst {
public:
    using IntConst::IntConst;

    static bool classof(Attribute attr)
    {
        const auto cst = llvm::dyn_cast<IntConst>(attr);
        if (!cst) return false;
        const auto splat = cst.tryGetSplatValue();
        if (!splat) return false;
        return splat->isAllOnes();
    }
};

} // namespace

using SelectFolder = FoldingContext<arith::SelectOp>;
static constexpr auto selectFolder =
    match::AssembledMatcher<SelectFolder>()
        .match([](Value, Value trueValue, Value falseValue) -> OpFoldResult {
            if (trueValue == falseValue) return trueValue;
            return {};
        })
        .match([](Zeros, Value, Value falseValue) -> OpFoldResult {
            return falseValue;
        })
        .match([](Ones, Value trueValue, Value) -> OpFoldResult {
            return trueValue;
        })
        .match([](PoisonAttr, Type opTy, Value) -> OpFoldResult {
            return PoisonAttr::get(opTy);
        })
        .match(
            [](IntConst cond,
               IntConst trueValue,
               IntConst falseValue) -> OpFoldResult {
                return cond.map(
                    trueValue.getElementType(),
                    [](auto c, auto t, auto f) -> std::optional<llvm::APInt> {
                        if (!c) return std::nullopt;
                        return c->isAllOnes() ? t : f;
                    },
                    trueValue,
                    falseValue);
            });

static FailureOr<OpFoldResult> simulateFold(arith::SelectOp op)
{
    SmallVector<Attribute, 3> operands;
    for (auto op : op->getOperands())
        matchPattern(op, m_Constant(&operands.emplace_back()));

    SmallVector<OpFoldResult, 1> results;
    if (auto res = selectFolder(make_folding_context(op, operands, results))) {
        if (succeeded(*res)) return results.back();
    }
    return failure();
}

static std::string stringify(FailureOr<OpFoldResult> result)
{
    if (failed(result)) return "fail";
    if (auto constant = llvm::dyn_cast_if_present<IntConst>(
            result->dyn_cast<Attribute>())) {
        return constant.toStringZExt();
    }
    if (auto arg = llvm::dyn_cast<BlockArgument>(result->dyn_cast<Value>()))
        return Twine("arg").concat(Twine(arg.getArgNumber())).str();
    return "error";
}

TEST_CASE("FoldContext")
{
    MLIRContext ctx;
    ctx.loadDialect<ubx::UBXDialect>();
    ctx.loadDialect<arith::ArithDialect>();

    const auto i1 = IntegerType::get(&ctx, 1);
    const auto i64 = IntegerType::get(&ctx, 64);
    const auto v3_i1 = VectorType::get({3}, i1);
    const auto v3_i64 = VectorType::get({3}, i64);
    auto loc = UnknownLoc::get(&ctx);

    OpBuilder builder(&ctx);
    auto region = std::make_unique<Region>();
    auto block = builder.createBlock(
        region.get(),
        {},
        {v3_i1, v3_i64, v3_i64},
        {loc, loc, loc});
    auto args = block->getArguments();

    auto v3_f = DenseIntElementsAttr::get(v3_i1, llvm::APInt(1U, 0));
    auto v3_t = DenseIntElementsAttr::get(v3_i1, llvm::APInt(1U, 1));
    auto f = builder.create<arith::ConstantOp>(loc, v3_f).getResult();
    auto t = builder.create<arith::ConstantOp>(loc, v3_t).getResult();
    auto p = builder.create<ubx::PoisonOp>(loc, v3_i1).getResult();
    auto v3_0p1 = IntConst::get(
        v3_i1,
        {llvm::APInt(1U, 0), llvm::APInt(1U, 0), llvm::APInt(1U, 1)},
        {false, true, false});
    auto c =
        builder
            .create<ubx::PoisonOp>(loc, llvm::cast<PoisonAttrInterface>(v3_0p1))
            .getResult();

    // select(?, ?, ?) = ?
    auto sel_xyz =
        builder.create<arith::SelectOp>(loc, args[0], args[1], args[2]);
    CHECK_EQ(stringify(simulateFold(sel_xyz)), "fail");

    // select(?, x, x) = x
    auto sel_xyy =
        builder.create<arith::SelectOp>(loc, args[0], args[1], args[1]);
    CHECK_EQ(stringify(simulateFold(sel_xyy)), "arg1");

    // select(false, ?, x) = x
    auto sel_fxy = builder.create<arith::SelectOp>(loc, f, args[1], args[2]);
    CHECK_EQ(stringify(simulateFold(sel_fxy)), "arg2");

    // select(true, x, ?) = x
    auto sel_txy = builder.create<arith::SelectOp>(loc, t, args[1], args[2]);
    CHECK_EQ(stringify(simulateFold(sel_txy)), "arg1");

    // select(psn, x, y) = psn
    auto sel_pxy = builder.create<arith::SelectOp>(loc, p, args[1], args[2]);
    CHECK_EQ(stringify(simulateFold(sel_pxy)), "[psn, psn, psn]");

    // select([0, psn, 1], [1, 1, 1], [0, 0, 0]) = [0, psn, 1]
    auto sel_c01 = builder.create<arith::SelectOp>(loc, c, t, f);
    CHECK_EQ(stringify(simulateFold(sel_c01)), "[0, psn, 1]");
}
