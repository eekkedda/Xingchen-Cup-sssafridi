// MhcExpand forward/backward kernel.
#include "kernel_operator.h"

#include "mhc_expand_tiling.h"
#include "tiling_key_mhc_expand.h"

namespace {
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint64_t MAX_UINT16 = 0xFFFFULL;
}

template <class DT_X, int IS_BACKWARD>
class KernelMhcExpand {
public:
    __aicore__ inline explicit KernelMhcExpand(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR o, const MhcExpandTilingData &tiling) {
        s_ = tiling.s;
        d_ = tiling.d;
        mhcMult_ = tiling.mhcMult;
        backward_ = (IS_BACKWARD != 0);
        tileLength_ = tiling.tileLength;
        rowBlock_ = tiling.rowBlock;
        blockDim_ = tiling.blockDim;
        tilesPerRow_ = (d_ + tileLength_ - 1) / tileLength_;
        rowGroups_ = (s_ + rowBlock_ - 1) / rowBlock_;
        tileElements_ = tileLength_ * rowBlock_;

        const uint64_t inputLength = backward_ ? s_ * mhcMult_ * d_ : s_ * d_;
        const uint64_t outputLength = backward_ ? s_ * d_ : s_ * mhcMult_ * d_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputLength);
        oGm_.SetGlobalBuffer((__gm__ DT_X *)o, outputLength);

        if (backward_) {
            pipe_->InitBuffer(inQueue_, 2, tileElements_ * sizeof(DT_X));
            pipe_->InitBuffer(outQueue_, 1, tileElements_ * sizeof(DT_X));
            pipe_->InitBuffer(valueFp32Buf_, tileElements_ * sizeof(float));
            pipe_->InitBuffer(accFp32Buf_, tileElements_ * sizeof(float));
        } else {
            // Ping-pong buffers overlap the next MTE2 read with the current
            // tile's MTE3 replica writes.
            pipe_->InitBuffer(forwardBuf_, 2 * tileElements_ * sizeof(DT_X));
        }
    }

    __aicore__ inline void Process() {
        const uint64_t taskCount = rowGroups_ * tilesPerRow_;
        if (!backward_) {
            if (taskCount <= blockDim_) {
                ProcessForwardSinglePass(taskCount);
            } else {
                ProcessForwardTasks(taskCount);
            }
            return;
        }

        for (uint64_t task = AscendC::GetBlockIdx(); task < taskCount; task += blockDim_) {
            const uint64_t rowGroup = task / tilesPerRow_;
            const uint64_t rowStart = rowGroup * rowBlock_;
            const uint16_t rowCount = static_cast<uint16_t>(
                ((s_ - rowStart) < rowBlock_) ? (s_ - rowStart) : rowBlock_);
            const uint64_t columnOffset = (task % tilesPerRow_) * tileLength_;
            const uint32_t validLength = static_cast<uint32_t>(
                ((d_ - columnOffset) < tileLength_) ? (d_ - columnOffset) : tileLength_);
            const uint32_t alignedLength = AlignUp(validLength);
            ProcessBackward(rowStart, rowCount, columnOffset, validLength, alignedLength);
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t length) const {
        constexpr uint32_t elementsPerBlock = BLOCK_BYTES / sizeof(DT_X);
        return (length + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
    }

    __aicore__ inline bool CanUseAlignedCopy(uint64_t offset, uint64_t gmRowStride,
                                              uint32_t validLength) const {
        constexpr uint32_t elementsPerBlock = BLOCK_BYTES / sizeof(DT_X);
        const uint64_t gapBlocks = (gmRowStride - validLength) / elementsPerBlock;
        return (offset % elementsPerBlock == 0) &&
               (validLength % elementsPerBlock == 0) &&
               (gmRowStride % elementsPerBlock == 0) && gapBlocks <= MAX_UINT16;
    }

    __aicore__ inline void CopyRowsIn(const AscendC::LocalTensor<DT_X> &dst,
                                      uint64_t gmOffset, uint16_t rowCount,
                                      uint64_t gmRowStride, uint32_t validLength,
                                      uint32_t alignedLength) {
        constexpr uint32_t elementsPerBlock = BLOCK_BYTES / sizeof(DT_X);
        if (CanUseAlignedCopy(gmOffset, gmRowStride, validLength)) {
            const uint16_t blockLen = static_cast<uint16_t>(validLength / elementsPerBlock);
            const uint16_t srcGap = static_cast<uint16_t>(
                (gmRowStride - validLength) / elementsPerBlock);
            AscendC::DataCopyParams copyParams{rowCount, blockLen, srcGap, 0};
            AscendC::DataCopy(dst, xGm_[gmOffset], copyParams);
        } else {
            const uint32_t copyBytes = validLength * static_cast<uint32_t>(sizeof(DT_X));
            const uint32_t srcGapBytes = static_cast<uint32_t>(
                (gmRowStride - validLength) * sizeof(DT_X));
            AscendC::DataCopyExtParams copyParams{
                rowCount, copyBytes, srcGapBytes, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{
                true, 0, static_cast<uint8_t>(alignedLength - validLength), static_cast<DT_X>(0)};
            AscendC::DataCopyPad(dst, xGm_[gmOffset], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyRowsOut(uint64_t gmOffset,
                                       const AscendC::LocalTensor<DT_X> &src,
                                       uint16_t rowCount, uint64_t gmRowStride,
                                       uint32_t validLength) {
        constexpr uint32_t elementsPerBlock = BLOCK_BYTES / sizeof(DT_X);
        if (CanUseAlignedCopy(gmOffset, gmRowStride, validLength)) {
            const uint16_t blockLen = static_cast<uint16_t>(validLength / elementsPerBlock);
            const uint16_t dstGap = static_cast<uint16_t>(
                (gmRowStride - validLength) / elementsPerBlock);
            AscendC::DataCopyParams copyParams{rowCount, blockLen, 0, dstGap};
            AscendC::DataCopy(oGm_[gmOffset], src, copyParams);
        } else {
            const uint32_t copyBytes = validLength * static_cast<uint32_t>(sizeof(DT_X));
            const uint32_t dstGapBytes = static_cast<uint32_t>(
                (gmRowStride - validLength) * sizeof(DT_X));
            AscendC::DataCopyExtParams copyParams{
                rowCount, copyBytes, 0, dstGapBytes, 0};
            AscendC::DataCopyPad(oGm_[gmOffset], src, copyParams);
        }
    }

    __aicore__ inline void CopyBackwardIn(uint64_t gmOffset, uint16_t rowCount,
                                          uint32_t validLength, uint32_t alignedLength) {
        AscendC::LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
        CopyRowsIn(xLocal, gmOffset, rowCount, mhcMult_ * d_, validLength, alignedLength);
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CastFromFloat(AscendC::LocalTensor<DT_X> &dst,
                                         const AscendC::LocalTensor<float> &src,
                                         uint32_t length) {
        if constexpr (AscendC::IsSameType<DT_X, bfloat16_t>::value) {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, length);
        } else {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, length);
        }
    }

    __aicore__ inline void ProcessForwardTasks(uint64_t taskCount) {
        AscendC::LocalTensor<DT_X> forwardLocal = forwardBuf_.Get<DT_X>();
        const event_t mte2ToMte3[2] = {
            static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3)),
            static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3))};
        const event_t mte3ToMte2[2] = {
            static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2)),
            static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2))};
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2[1]);

        uint32_t slot = 0;
        for (uint64_t task = AscendC::GetBlockIdx(); task < taskCount; task += blockDim_) {
            const uint64_t rowGroup = task / tilesPerRow_;
            const uint64_t rowStart = rowGroup * rowBlock_;
            const uint16_t rowCount = static_cast<uint16_t>(
                ((s_ - rowStart) < rowBlock_) ? (s_ - rowStart) : rowBlock_);
            const uint64_t columnOffset = (task % tilesPerRow_) * tileLength_;
            const uint32_t validLength = static_cast<uint32_t>(
                ((d_ - columnOffset) < tileLength_) ? (d_ - columnOffset) : tileLength_);
            const uint32_t alignedLength = AlignUp(validLength);
            AscendC::LocalTensor<DT_X> xLocal = forwardLocal[slot * tileElements_];
            const uint64_t inputOffset = rowStart * d_ + columnOffset;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2[slot]);
            CopyRowsIn(xLocal, inputOffset, rowCount, d_, validLength, alignedLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte2ToMte3[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte2ToMte3[slot]);

            for (uint32_t replica = 0; replica < mhcMult_; ++replica) {
                const uint64_t outputOffset =
                    (rowStart * mhcMult_ + replica) * d_ + columnOffset;
                CopyRowsOut(outputOffset, xLocal, rowCount, mhcMult_ * d_, validLength);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2[slot]);
            slot ^= 1U;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2[1]);
    }

    __aicore__ inline void ProcessForwardSinglePass(uint64_t taskCount) {
        const uint64_t task = AscendC::GetBlockIdx();
        if (task >= taskCount) {
            return;
        }
        AscendC::LocalTensor<DT_X> xLocal = forwardBuf_.Get<DT_X>();
        const uint64_t rowGroup = task / tilesPerRow_;
        const uint64_t rowStart = rowGroup * rowBlock_;
        const uint16_t rowCount = static_cast<uint16_t>(
            ((s_ - rowStart) < rowBlock_) ? (s_ - rowStart) : rowBlock_);
        const uint64_t columnOffset = (task % tilesPerRow_) * tileLength_;
        const uint32_t validLength = static_cast<uint32_t>(
            ((d_ - columnOffset) < tileLength_) ? (d_ - columnOffset) : tileLength_);
        const uint32_t alignedLength = AlignUp(validLength);
        const uint64_t inputOffset = rowStart * d_ + columnOffset;

        CopyRowsIn(xLocal, inputOffset, rowCount, d_, validLength, alignedLength);
        const event_t mte2ToMte3 = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte2ToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte2ToMte3);
        for (uint32_t replica = 0; replica < mhcMult_; ++replica) {
            const uint64_t outputOffset =
                (rowStart * mhcMult_ + replica) * d_ + columnOffset;
            CopyRowsOut(outputOffset, xLocal, rowCount, mhcMult_ * d_, validLength);
        }
    }

    __aicore__ inline void ProcessBackward(uint64_t rowStart, uint16_t rowCount,
                                            uint64_t columnOffset, uint32_t validLength,
                                            uint32_t alignedLength) {
        const uint32_t computeLength = static_cast<uint32_t>(rowCount) * alignedLength;
        AscendC::LocalTensor<float> valueFp32 = valueFp32Buf_.Get<float>();
        AscendC::LocalTensor<float> accumulator = accFp32Buf_.Get<float>();
        const uint64_t baseOffset = rowStart * mhcMult_ * d_ + columnOffset;
        CopyBackwardIn(baseOffset, rowCount, validLength, alignedLength);

        for (uint32_t replica = 0; replica < mhcMult_; ++replica) {
            AscendC::LocalTensor<DT_X> xLocal = inQueue_.DeQue<DT_X>();
            if (replica + 1 < mhcMult_) {
                CopyBackwardIn(baseOffset + (replica + 1) * d_, rowCount,
                               validLength, alignedLength);
            }
            if (replica == 0) {
                AscendC::Cast(accumulator, xLocal, AscendC::RoundMode::CAST_NONE, computeLength);
            } else if constexpr (AscendC::IsSameType<DT_X, half>::value) {
                AscendC::Axpy(accumulator, xLocal, static_cast<DT_X>(1.0F),
                              static_cast<int32_t>(computeLength));
            } else {
                AscendC::Cast(valueFp32, xLocal, AscendC::RoundMode::CAST_NONE, computeLength);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumulator, accumulator, valueFp32, computeLength);
            }
            AscendC::PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(xLocal);
        }

        AscendC::LocalTensor<DT_X> oLocal = outQueue_.AllocTensor<DT_X>();
        CastFromFloat(oLocal, accumulator, computeLength);
        outQueue_.EnQue(oLocal);
        oLocal = outQueue_.DeQue<DT_X>();
        CopyRowsOut(rowStart * d_ + columnOffset, oLocal, rowCount, d_, validLength);
        outQueue_.FreeTensor(oLocal);
    }

    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> valueFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> forwardBuf_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> oGm_;
    uint64_t s_ = 0;
    uint64_t d_ = 0;
    uint64_t tilesPerRow_ = 0;
    uint64_t rowGroups_ = 0;
    uint32_t mhcMult_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t tileElements_ = 0;
    uint32_t rowBlock_ = 0;
    uint32_t blockDim_ = 0;
    bool backward_ = false;
};

template <typename DT_X, int IS_BACKWARD>
__global__ __aicore__ void mhc_expand(GM_ADDR x, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcExpandTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcExpandTilingData, tiling_data, tiling);
    AscendC::TPipe pipe;
    KernelMhcExpand<DT_X, IS_BACKWARD> op(&pipe);
    op.Init(x, o, tiling_data);
    op.Process();
}
