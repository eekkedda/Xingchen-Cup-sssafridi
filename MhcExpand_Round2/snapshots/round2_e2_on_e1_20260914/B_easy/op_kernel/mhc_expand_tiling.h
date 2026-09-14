// Host-to-kernel tiling data for MhcExpand.
#pragma once

#include <cstdint>

struct MhcExpandTilingData {
    uint64_t s;
    uint64_t d;
    uint32_t mhcMult;
    uint32_t backward;
    uint32_t tileLength;
    uint32_t rowBlock;
    uint32_t blockDim;
};
