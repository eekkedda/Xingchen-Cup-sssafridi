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
            // Raw double-buffered input plus one output buffer; same total UB
            // bytes and allocation order as the previous TQue layout
            // (depth-2 VECIN + depth-1 VECOUT).
            pipe_->InitBuffer(backwardBuf_, 2 * tileElements_ * sizeof(DT_X));
            pipe_->InitBuffer(outBuf_, tileElements_ * sizeof(DT_X));
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

        ProcessBackwardTasks(taskCount);
    }

private:
    static constexpr bool kBackward_ = (IS_BACKWARD != 0);

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

    __aicore__ inline void CastFromFloat(AscendC::LocalTensor<DT_X> &dst,
                                         const AscendC::LocalTensor<float> &src,
                                         uint32_t length) {
        if constexpr (AscendC::IsSameType<DT_X, bfloat16_t>::value) {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, length);
        } else {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, length);
        }
    }

    // Backward pipeline over a raw double-buffered input and explicit events,
    // mirroring the proven forward ping-pong across the MTE2 -> Vector -> MTE3
    // chain. Each Set/Wait pair reproduces the exact TQue dependency it
    // replaces (VECIN: EnQue=MTE2_V, Alloc waits V_MTE2; VECOUT:
    // EnQue=V_MTE3, Alloc waits MTE3_V) without the queue bookkeeping.
    __aicore__ inline void ProcessBackwardTasks(uint64_t taskCount) {
        AscendC::LocalTensor<DT_X> inLocal = backwardBuf_.Get<DT_X>();
        AscendC::LocalTensor<DT_X> outLocal = outBuf_.Get<DT_X>();
        AscendC::LocalTensor<float> accumulator = accFp32Buf_.Get<float>();
        AscendC::LocalTensor<float> valueFp32 = valueFp32Buf_.Get<float>();
        const event_t mte2ToV[2] = {
            static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V)),
            static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V))};
        const event_t vToMte2[2] = {
            static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2)),
            static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2))};
        const event_t vToMte3 = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        const event_t mte3ToV = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2[1]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);

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
            const uint32_t computeLength = static_cast<uint32_t>(rowCount) * alignedLength;
            const uint64_t baseOffset = rowStart * mhcMult_ * d_ + columnOffset;

            // Issue the replica-0 read into the current slot.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2[slot]);
            CopyRowsIn(inLocal[slot * tileElements_], baseOffset, rowCount, mhcMult_ * d_,
                       validLength, alignedLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV[slot]);

            for (uint32_t replica = 0; replica < mhcMult_; ++replica) {
                const uint32_t next = slot ^ 1U;
                if (replica + 1 < mhcMult_) {
                    // Prefetch the next replica while the current one computes.
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2[next]);
                    CopyRowsIn(inLocal[next * tileElements_],
                               baseOffset + (replica + 1) * d_, rowCount, mhcMult_ * d_,
                               validLength, alignedLength);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV[next]);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV[slot]);
                AscendC::LocalTensor<DT_X> xLocal = inLocal[slot * tileElements_];
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
                // The Vector pipe is done with this slot; MTE2 may overwrite it.
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2[slot]);
                slot = next;
            }

            // Output: wait until MTE3 finished reading the previous task's
            // result, cast, then hand the buffer to MTE3.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
            CastFromFloat(outLocal, accumulator, computeLength);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
            CopyRowsOut(rowStart * d_ + columnOffset, outLocal, rowCount, d_, validLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
        }
        // Drain: the final MTE3 read of outLocal must complete before the
        // kernel exits and the TPipe tears down.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
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

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> backwardBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf_;
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
