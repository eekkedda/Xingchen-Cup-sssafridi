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
        tileLength_ = tiling.tileLength;
        rowBlock_ = tiling.rowBlock;
        blockDim_ = tiling.blockDim;
        tilesPerRow_ = (d_ + tileLength_ - 1) / tileLength_;
        rowGroups_ = (s_ + rowBlock_ - 1) / rowBlock_;
        tileElements_ = tileLength_ * rowBlock_;

        // Direction is a compile-time template argument selected by the host
        // tiling key; the unused direction's code is eliminated at compile time.
        if constexpr (kBackward_) {
            const uint64_t inputLength = s_ * mhcMult_ * d_;
            const uint64_t outputLength = s_ * d_;
            xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputLength);
            oGm_.SetGlobalBuffer((__gm__ DT_X *)o, outputLength);
            // Grouped paths: mode 1 slots hold a whole m*D token block,
            // mode 2 slots hold a 2*D stream pair; rowBlock is 1 in both,
            // so tileElements_ equals d_ there.
            bwdGroupedMode_ = tiling.bwdGrouped;
            const uint32_t slotElements =
                bwdGroupedMode_ == 1 ? mhcMult_ * tileLength_
                : bwdGroupedMode_ == 2 ? 2 * tileLength_
                                       : tileElements_;
            pipe_->InitBuffer(inQueue_, 2, slotElements * sizeof(DT_X));
            pipe_->InitBuffer(outQueue_, 1, tileElements_ * sizeof(DT_X));
            pipe_->InitBuffer(valueFp32Buf_, tileElements_ * sizeof(float));
            pipe_->InitBuffer(accFp32Buf_, tileElements_ * sizeof(float));
        } else {
            const uint64_t inputLength = s_ * d_;
            const uint64_t outputLength = s_ * mhcMult_ * d_;
            xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputLength);
            oGm_.SetGlobalBuffer((__gm__ DT_X *)o, outputLength);
            // Ping-pong buffers overlap the next MTE2 read with the current
            // tile's MTE3 replica writes.
            pipe_->InitBuffer(forwardBuf_, 2 * tileElements_ * sizeof(DT_X));
        }
    }

    __aicore__ inline void Process() {
        const uint64_t taskCount = rowGroups_ * tilesPerRow_;
        if constexpr (!kBackward_) {
            if (taskCount <= blockDim_) {
                ProcessForwardSinglePass(taskCount);
            } else {
                ProcessForwardTasks(taskCount);
            }
            return;
        }

        if (bwdGroupedMode_ == 1) {
            ProcessBackwardGrouped(taskCount);
            return;
        }
        if (bwdGroupedMode_ == 2) {
            ProcessBackwardPaired(taskCount);
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
    static constexpr bool kBackward_ = (IS_BACKWARD != 0);

    // Pair-grouped backward path (mode 2): when the whole m*D block exceeds
    // the UB budget but a 2*D pair fits (large-D family), streams are fetched
    // two per 2D-block DMA in original k order; the odd tail group reads a
    // single stream. Same TQue cadence and accumulation order as mode 1.
    __aicore__ inline void ProcessBackwardPaired(uint64_t taskCount) {
        AscendC::LocalTensor<float> valueFp32 = valueFp32Buf_.Get<float>();
        AscendC::LocalTensor<float> accumulator = accFp32Buf_.Get<float>();
        const uint32_t computeLength = static_cast<uint32_t>(d_);
        constexpr uint32_t elementsPerBlock = BLOCK_BYTES / sizeof(DT_X);
        const uint16_t pairBlockLen =
            static_cast<uint16_t>(2 * computeLength / elementsPerBlock);
        const uint16_t singleBlockLen =
            static_cast<uint16_t>(computeLength / elementsPerBlock);
        const AscendC::DataCopyParams pairCopy{1, pairBlockLen, 0, 0};
        const AscendC::DataCopyParams singleCopy{1, singleBlockLen, 0, 0};
        for (uint64_t task = AscendC::GetBlockIdx(); task < taskCount; task += blockDim_) {
            const uint64_t rowBase = task * mhcMult_ * d_;
            uint32_t replica = 0;
            while (replica < mhcMult_) {
                const uint32_t streams =
                    (mhcMult_ - replica >= 2) ? 2U : 1U;
                AscendC::LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
                AscendC::DataCopy(xLocal, xGm_[rowBase + replica * d_],
                                  streams == 2 ? pairCopy : singleCopy);
                inQueue_.EnQue(xLocal);
                xLocal = inQueue_.DeQue<DT_X>();
                for (uint32_t s = 0; s < streams; ++s) {
                    if (replica == 0 && s == 0) {
                        AscendC::Cast(accumulator, xLocal,
                                      AscendC::RoundMode::CAST_NONE, computeLength);
                    } else if constexpr (AscendC::IsSameType<DT_X, half>::value) {
                        AscendC::Axpy(accumulator, xLocal[s * d_], static_cast<DT_X>(1.0F),
                                      static_cast<int32_t>(computeLength));
                    } else {
                        AscendC::Cast(valueFp32, xLocal[s * d_],
                                      AscendC::RoundMode::CAST_NONE, computeLength);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(accumulator, accumulator, valueFp32, computeLength);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                inQueue_.FreeTensor(xLocal);
                replica += streams;
            }

            AscendC::LocalTensor<DT_X> oLocal = outQueue_.AllocTensor<DT_X>();
            CastFromFloat(oLocal, accumulator, computeLength);
            outQueue_.EnQue(oLocal);
            oLocal = outQueue_.DeQue<DT_X>();
            CopyRowsOut(task * d_, oLocal, 1, d_, computeLength);
            outQueue_.FreeTensor(oLocal);
        }
    }

    // Grouped backward path (plan D on the TQue base): one 2D-block DMA per
    // token reads the m contiguous GM streams into a single TQue slot;
    // rowBlock==1 keeps replica k at offset k*D, contiguous and 16-aligned,
    // so the per-replica Cast/Axpy sequence is unchanged. The queue
    // Alloc/EnQue/DeQue/Free cadence mirrors the regular backward path.
    __aicore__ inline void ProcessBackwardGrouped(uint64_t taskCount) {
        AscendC::LocalTensor<float> valueFp32 = valueFp32Buf_.Get<float>();
        AscendC::LocalTensor<float> accumulator = accFp32Buf_.Get<float>();
        const uint32_t computeLength = static_cast<uint32_t>(d_);
        const uint32_t slotElements = mhcMult_ * static_cast<uint32_t>(d_);
        constexpr uint32_t elementsPerBlock = BLOCK_BYTES / sizeof(DT_X);
        const uint16_t blockLen =
            static_cast<uint16_t>(slotElements / elementsPerBlock);
        const AscendC::DataCopyParams groupedCopy{1, blockLen, 0, 0};
        for (uint64_t task = AscendC::GetBlockIdx(); task < taskCount; task += blockDim_) {
            AscendC::LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
            AscendC::DataCopy(xLocal, xGm_[task * mhcMult_ * d_], groupedCopy);
            inQueue_.EnQue(xLocal);
            xLocal = inQueue_.DeQue<DT_X>();

            AscendC::Cast(accumulator, xLocal, AscendC::RoundMode::CAST_NONE, computeLength);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t replica = 1; replica < mhcMult_; ++replica) {
                if constexpr (AscendC::IsSameType<DT_X, half>::value) {
                    AscendC::Axpy(accumulator, xLocal[replica * d_], static_cast<DT_X>(1.0F),
                                  static_cast<int32_t>(computeLength));
                } else {
                    AscendC::Cast(valueFp32, xLocal[replica * d_],
                                  AscendC::RoundMode::CAST_NONE, computeLength);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(accumulator, accumulator, valueFp32, computeLength);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            inQueue_.FreeTensor(xLocal);

            AscendC::LocalTensor<DT_X> oLocal = outQueue_.AllocTensor<DT_X>();
            CastFromFloat(oLocal, accumulator, computeLength);
            outQueue_.EnQue(oLocal);
            oLocal = outQueue_.DeQue<DT_X>();
            CopyRowsOut(task * d_, oLocal, 1, d_, computeLength);
            outQueue_.FreeTensor(oLocal);
        }
    }

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
    uint32_t bwdGroupedMode_ = 0;
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
