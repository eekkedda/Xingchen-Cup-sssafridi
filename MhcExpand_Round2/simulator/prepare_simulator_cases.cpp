#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../../B_easy/op_kernel/mhc_expand_tiling.h"

namespace {
constexpr uint64_t S = 1;
constexpr uint64_t D = 16;
constexpr uint32_t M = 2;
constexpr uint64_t S_M8 = 1;
constexpr uint64_t D_M8 = 256;
constexpr uint32_t M_M8 = 8;

template <typename T>
void WriteBinary(const std::string &path, const T &value) {
    static_assert(std::is_trivially_copyable<T>::value, "binary fixture must be POD");
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
    if (!stream) {
        throw std::runtime_error("cannot write " + path);
    }
}

template <typename T>
void WriteVector(const std::string &path, const std::vector<T> &values) {
    static_assert(std::is_trivially_copyable<T>::value, "binary fixture must be POD");
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    stream.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!stream) {
        throw std::runtime_error("cannot write " + path);
    }
}

uint16_t FloatToBfloat16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

MhcExpandTilingData MakeTiling(bool backward) {
    MhcExpandTilingData tiling{};
    tiling.s = S;
    tiling.d = D;
    tiling.mhcMult = M;
    tiling.backward = backward ? 1U : 0U;
    tiling.tileLength = 16;
    tiling.rowBlock = 1;
    tiling.blockDim = 1;
    tiling.tilesPerRow = 1;
    tiling.taskCount = 1;
    return tiling;
}

MhcExpandTilingData MakeBackwardM8Tiling() {
    MhcExpandTilingData tiling{};
    tiling.s = S_M8;
    tiling.d = D_M8;
    tiling.mhcMult = M_M8;
    tiling.backward = 1;
    tiling.tileLength = D_M8;
    tiling.rowBlock = 1;
    tiling.blockDim = 1;
    tiling.tilesPerRow = 1;
    tiling.taskCount = 1;
    return tiling;
}
}  // namespace

int main() {
    static_assert(sizeof(_Float16) == sizeof(uint16_t), "host _Float16 must be IEEE binary16");
    static_assert(sizeof(MhcExpandTilingData) == 48, "unexpected tiling ABI size");

    std::vector<_Float16> forwardInput(D);
    for (uint64_t d = 0; d < D; ++d) {
        forwardInput[d] = static_cast<_Float16>(d + 1);
    }
    std::vector<_Float16> forwardGolden(S * M * D);
    for (uint32_t replica = 0; replica < M; ++replica) {
        for (uint64_t d = 0; d < D; ++d) {
            forwardGolden[replica * D + d] = forwardInput[d];
        }
    }

    std::vector<_Float16> backwardInput(S * M * D);
    std::vector<_Float16> backwardGolden(S * D);
    for (uint64_t d = 0; d < D; ++d) {
        backwardInput[d] = static_cast<_Float16>(d + 1);
        backwardInput[D + d] = static_cast<_Float16>(0.5);
        backwardGolden[d] = static_cast<_Float16>(static_cast<float>(d + 1) + 0.5F);
    }

    WriteBinary("forward_tiling.bin", MakeTiling(false));
    WriteBinary("backward_tiling.bin", MakeTiling(true));
    WriteVector("forward_input.bin", forwardInput);
    WriteVector("forward_golden.bin", forwardGolden);
    WriteVector("backward_input.bin", backwardInput);
    WriteVector("backward_golden.bin", backwardGolden);

    std::vector<uint16_t> forwardBfloatInput(D);
    std::vector<uint16_t> forwardBfloatGolden(S * M * D);
    std::vector<uint16_t> backwardBfloatInput(S * M * D);
    std::vector<uint16_t> backwardBfloatGolden(S * D);
    for (uint64_t d = 0; d < D; ++d) {
        forwardBfloatInput[d] = FloatToBfloat16(static_cast<float>(d + 1));
        backwardBfloatInput[d] = forwardBfloatInput[d];
        backwardBfloatInput[D + d] = FloatToBfloat16(0.5F);
        backwardBfloatGolden[d] = FloatToBfloat16(static_cast<float>(d + 1) + 0.5F);
    }
    for (uint32_t replica = 0; replica < M; ++replica) {
        for (uint64_t d = 0; d < D; ++d) {
            forwardBfloatGolden[replica * D + d] = forwardBfloatInput[d];
        }
    }
    WriteVector("forward_bfloat16_input.bin", forwardBfloatInput);
    WriteVector("forward_bfloat16_golden.bin", forwardBfloatGolden);
    WriteVector("backward_bfloat16_input.bin", backwardBfloatInput);
    WriteVector("backward_bfloat16_golden.bin", backwardBfloatGolden);

    std::vector<_Float16> backwardM8Input(S_M8 * M_M8 * D_M8);
    std::vector<_Float16> backwardM8Golden(S_M8 * D_M8);
    for (uint64_t d = 0; d < D_M8; ++d) {
        float sum = 0.0F;
        for (uint32_t replica = 0; replica < M_M8; ++replica) {
            const float value = static_cast<float>(replica + 1) * 0.125F;
            backwardM8Input[replica * D_M8 + d] = static_cast<_Float16>(value);
            sum += value;
        }
        backwardM8Golden[d] = static_cast<_Float16>(sum);
    }
    WriteBinary("backward_m8_tiling.bin", MakeBackwardM8Tiling());
    WriteVector("backward_m8_input.bin", backwardM8Input);
    WriteVector("backward_m8_golden.bin", backwardM8Golden);

    // E3 boundary case A: D > MAX_TILE_ELEMENTS forces tilesPerRow == 2 and
    // exercises the generic divide/modulo task path plus aligned copies with
    // a nonzero column offset. Host-derived tiling for S=3, D=8208, m=2:
    // tileLength=8192, rowBlock=1, blockDim=6, taskCount=6.
    constexpr uint64_t S_A = 3;
    constexpr uint64_t D_A = 8208;
    constexpr uint32_t M_A = 2;
    MhcExpandTilingData tilingA{};
    tilingA.s = S_A;
    tilingA.d = D_A;
    tilingA.mhcMult = M_A;
    tilingA.backward = 0;
    tilingA.tileLength = 8192;
    tilingA.rowBlock = 1;
    tilingA.blockDim = 6;
    tilingA.tilesPerRow = 2;
    tilingA.taskCount = 6;
    std::vector<_Float16> forwardAInput(S_A * D_A);
    std::vector<_Float16> forwardAGolden(S_A * M_A * D_A);
    for (uint64_t i = 0; i < S_A * D_A; ++i) {
        forwardAInput[i] = static_cast<_Float16>((i % 1000u) * 0.5F);
    }
    for (uint64_t i = 0; i < S_A; ++i) {
        for (uint32_t replica = 0; replica < M_A; ++replica) {
            for (uint64_t d = 0; d < D_A; ++d) {
                forwardAGolden[(i * M_A + replica) * D_A + d] = forwardAInput[i * D_A + d];
            }
        }
    }
    WriteBinary("forward_d8208_tiling.bin", tilingA);
    WriteVector("forward_d8208_input.bin", forwardAInput);
    WriteVector("forward_d8208_golden.bin", forwardAGolden);

    // E3 boundary case B: non-16-aligned D with tilesPerRow == 1 exercises the
    // single-tile fast path together with DataCopyPad. Values keep the FP32
    // accumulation and final FP16 conversion exact. Host-derived tiling for
    // S=3, D=17, m=4: tileLength=32, rowBlock=1, blockDim=3, taskCount=3.
    constexpr uint64_t S_B = 3;
    constexpr uint64_t D_B = 17;
    constexpr uint32_t M_B = 4;
    MhcExpandTilingData tilingB{};
    tilingB.s = S_B;
    tilingB.d = D_B;
    tilingB.mhcMult = M_B;
    tilingB.backward = 1;
    tilingB.tileLength = 32;
    tilingB.rowBlock = 1;
    tilingB.blockDim = 3;
    tilingB.tilesPerRow = 1;
    tilingB.taskCount = 3;
    std::vector<_Float16> backwardBInput(S_B * M_B * D_B);
    std::vector<_Float16> backwardBGolden(S_B * D_B);
    for (uint64_t i = 0; i < S_B; ++i) {
        for (uint64_t d = 0; d < D_B; ++d) {
            float sum = 0.0F;
            for (uint32_t replica = 0; replica < M_B; ++replica) {
                const float value = static_cast<float>(d + 1) +
                                    static_cast<float>(replica) * 0.25F;
                backwardBInput[(i * M_B + replica) * D_B + d] = static_cast<_Float16>(value);
                sum += value;
            }
            backwardBGolden[i * D_B + d] = static_cast<_Float16>(sum);
        }
    }
    WriteBinary("backward_d17_tiling.bin", tilingB);
    WriteVector("backward_d17_input.bin", backwardBInput);
    WriteVector("backward_d17_golden.bin", backwardBGolden);
    return 0;
}
