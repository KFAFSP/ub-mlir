#include "ub-mlir/Dialect/UB/Utils/StaticFolder.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <doctest/doctest.h>

using namespace mlir;
using namespace mlir::ub;

static const StaticFolder folder1([](Operation* op,
                                     ValueRange,
                                     SmallVectorImpl<OpFoldResult> &result) {
    result.emplace_back(UnitAttr::get(op->getContext()));
    return success();
});
static const StaticFolder folder2([](tensor::FromElementsOp op, ValueRange) {
    return op;
});
static const StaticFolder
    folder3([](tensor::FromElementsOp op, Value, IntegerAttr) { return op; });

TEST_CASE("StaticFolder")
{
    DialectRegistry registry;
    registry.insert<tensor::TensorDialect>();
    MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();

    const auto loc = mlir::UnknownLoc::get(&ctx);
    auto module = mlir::ModuleOp::create(loc);
    OpBuilder builder(module.getBodyRegion());

    const auto i64Ty = builder.getI64Type();
    auto emptyOp = builder.create<tensor::EmptyOp>(
        loc,
        ArrayRef<std::int64_t>{3, 3},
        i64Ty);
    auto dimOp = builder.create<tensor::DimOp>(loc, emptyOp.getResult(), 0);
    auto fromOp1 = builder.create<tensor::FromElementsOp>(
        loc,
        ValueRange{dimOp.getResult()});
    auto fromOp2 = builder.create<tensor::FromElementsOp>(
        loc,
        ValueRange{dimOp.getResult(), dimOp.getResult()});

    const auto none = Attribute{};

    CHECK(folder1(emptyOp, {}));
    CHECK(folder1(dimOp, {none, none}));
    CHECK(folder1(fromOp1, {none}));
    CHECK(folder1(fromOp2, {none, none}));

    CHECK(!folder2(emptyOp, {}));
    CHECK(!folder2(dimOp, {none, none}));
    CHECK(folder2(fromOp1, {none}));
    CHECK(folder2(fromOp2, {none, none}));

    CHECK(!folder3(emptyOp, {}));
    CHECK(!folder3(dimOp, {none, none}));
    CHECK(!folder3(fromOp1, {none}));
    CHECK(!folder3(fromOp2, {none, none}));
    CHECK(!folder3(fromOp2, {none, builder.getUnitAttr()}));
    CHECK(folder3(fromOp2, {none, builder.getI64IntegerAttr(1)}));
}
