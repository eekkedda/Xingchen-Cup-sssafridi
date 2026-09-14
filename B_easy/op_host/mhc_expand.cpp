// Host-side shape/type inference and tiling for MhcExpand.
#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_expand_tiling.h"
#include "../op_kernel/tiling_key_mhc_expand.h"

namespace optiling {
constexpr uint32_t MAX_TILE_ELEMENTS = 8192;
constexpr uint32_t MAX_ROWS_PER_TILE = 32;
constexpr uint32_t ELEMENTS_PER_BLOCK = 16;  // Both supported dtypes are 2 bytes.

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetInputDesc(0) == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t numCoresAiv = platform.GetCoreNumAiv();
    if (numCoresAiv <= 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *attrMhcMult = attrs->GetInt(0);
    const bool *attrBackward = attrs->GetBool(1);
    if (attrMhcMult == nullptr || attrBackward == nullptr || *attrMhcMult <= 0 ||
        static_cast<uint64_t>(*attrMhcMult) > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &inputShape = context->GetInputShape(0)->GetOriginShape();
    const size_t expectedRank = *attrBackward ? 3 : 2;
    if (inputShape.GetDimNum() != expectedRank) {
        return ge::GRAPH_FAILED;
    }

    const int64_t sDim = inputShape.GetDim(0);
    const int64_t dDim = inputShape.GetDim(*attrBackward ? 2 : 1);
    if (sDim <= 0 || dDim <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (*attrBackward && inputShape.GetDim(1) != *attrMhcMult) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t s = static_cast<uint64_t>(sDim);
    const uint64_t d = static_cast<uint64_t>(dDim);
    const uint64_t mhcMult = static_cast<uint64_t>(*attrMhcMult);
    if (d > std::numeric_limits<uint64_t>::max() / mhcMult ||
        s > std::numeric_limits<uint64_t>::max() / (d * mhcMult)) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t alignedD =
        (d + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
    const uint32_t tileLength =
        static_cast<uint32_t>(std::min<uint64_t>(MAX_TILE_ELEMENTS, alignedD));
    uint32_t rowBlock = static_cast<uint32_t>(std::min<uint64_t>(
        MAX_ROWS_PER_TILE, MAX_TILE_ELEMENTS / tileLength));
    // Multi-row DMA should reduce command count without collapsing a small
    // workload onto only one or two AIV cores.
    const uint64_t parallelRowBlock = std::max<uint64_t>(
        1, (s + static_cast<uint64_t>(numCoresAiv) - 1) /
               static_cast<uint64_t>(numCoresAiv));
    rowBlock = static_cast<uint32_t>(
        std::min<uint64_t>(rowBlock, parallelRowBlock));
    // Multi-row DataCopyExt encodes the GM row gap in uint32 bytes.
    if (d * mhcMult > std::numeric_limits<uint32_t>::max() / sizeof(uint16_t)) {
        rowBlock = 1;
    }
    const uint64_t tilesPerRow = (d + tileLength - 1) / tileLength;
    // Backward grouped-read paths: when one row fits a tile and D is
    // 16-element aligned, fetch contiguous streams with single 2D-block DMAs.
    // Full grouping (all m streams, one DMA per token) when the whole m*D
    // block fits the two TQue slots plus accumulator/output/staging budget;
    // otherwise pair grouping (two streams per DMA) when a 2*D pair fits.
    // Feasibility uses the platform-queried UB size; otherwise the regular
    // multi-row path runs.
    uint32_t bwdGroupedMode = 0;
    if (*attrBackward && tilesPerRow == 1 && d % ELEMENTS_PER_BLOCK == 0) {
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        const uint64_t fixedBytes =
            d * sizeof(float)     // FP32 accumulator
            + d * sizeof(uint16_t)  // output slot
            + d * sizeof(float);    // BF16 staging buffer
        const uint64_t fullBytes = 2ULL * mhcMult * d * sizeof(uint16_t);
        const uint64_t pairBytes = 2ULL * 2 * d * sizeof(uint16_t);
        if (fullBytes + fixedBytes <= ubSize) {
            bwdGroupedMode = 1;
            rowBlock = 1;
        } else if (mhcMult > 1 && pairBytes + fixedBytes <= ubSize) {
            bwdGroupedMode = 2;
            rowBlock = 1;
        }
    }
    const uint64_t rowGroups = (s + rowBlock - 1) / rowBlock;
    if (rowGroups > std::numeric_limits<uint64_t>::max() / tilesPerRow) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t taskCount = rowGroups * tilesPerRow;
    const uint32_t blockDim = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(numCoresAiv), taskCount));

    const ge::DataType dtypeX = context->GetInputDesc(0)->GetDataType();
    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    uint32_t IS_BACKWARD = *attrBackward ? 1U : 0U;
    ASCENDC_TPL_SEL_PARAM(context, DT_X, IS_BACKWARD);

    MhcExpandTilingData *tiling = context->GetTilingData<MhcExpandTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->s = s;
    tiling->d = d;
    tiling->mhcMult = static_cast<uint32_t>(*attrMhcMult);
    tiling->backward = *attrBackward ? 1U : 0U;
    tiling->tileLength = tileLength;
    tiling->rowBlock = rowBlock;
    tiling->blockDim = blockDim;
    tiling->bwdGrouped = bwdGroupedMode;

    context->SetBlockDim(blockDim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr || context->GetAttrs() == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    const int64_t *mhcMult = context->GetAttrs()->GetInt(0);
    const bool *backward = context->GetAttrs()->GetBool(1);
    if (mhcMult == nullptr || backward == nullptr || *mhcMult <= 0) {
        return GRAPH_FAILED;
    }

    if (*backward) {
        if (inputShape->GetDimNum() != 3 ||
            (inputShape->GetDim(1) >= 0 && inputShape->GetDim(1) != *mhcMult)) {
            return GRAPH_FAILED;
        }
        outputShape->SetDimNum(2);
        outputShape->SetDim(0, inputShape->GetDim(0));
        outputShape->SetDim(1, inputShape->GetDim(2));
    } else {
        if (inputShape->GetDimNum() != 2) {
            return GRAPH_FAILED;
        }
        outputShape->SetDimNum(3);
        outputShape->SetDim(0, inputShape->GetDim(0));
        outputShape->SetDim(1, *mhcMult);
        outputShape->SetDim(2, inputShape->GetDim(1));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MhcExpand : public OpDef {
public:
    explicit MhcExpand(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("o")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("mhc_mult").AttrType(OPTIONAL).Int(2);
        this->Attr("backward").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(MhcExpand);
}  // namespace ops
