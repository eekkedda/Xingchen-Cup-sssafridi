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
    // Backward grouped-read path flag (occupies the former 4-byte padding
    // slot; struct stays 40 bytes). 1 = full grouping: each token's m
    // contiguous GM streams are fetched with a single 2D-block DMA into one
    // TQue slot (rowBlock forced to 1). 2 = pair grouping: when the whole
    // m*D block exceeds the UB budget but a 2*D pair fits, streams are
    // fetched two at a time (odd tail group handled), halving read commands.
    uint32_t bwdGrouped;
};
static_assert(sizeof(MhcExpandTilingData) == 40, "unexpected tiling ABI size");
