#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""CuPy-compiled CUDA kernels for CTC, Zipformer, and Parakeet decoding.

Runtime validation bounds fixed engine tensors and configured search tables to
signed 32-bit indexing, while kernels clamp device-resident output lengths. The
search state remains on the GPU and is safe to replay from CUDA graphs; CTC
widens flattened offsets because its output stride comes directly from a dynamic
frame count. Packed copies also check pointer alignment: a contiguous view may
start inside an allocation. Nonfinite transducer candidates are treated as
unreachable before top-k selection. History fingerprints only filter possible
merges; matching token sequences are verified through their GPU backpointers.
"""

from functools import cache

import cupy as cp
import numpy as np

from ..constants import INT32_MAX, ZIPFORMER_BEAM_SEARCH_THREADS

CTC_COLLAPSE_KERNEL = cp.RawKernel(
    r"""
    extern "C" __global__ void ctc_collapse(const int* paths, const int* output_lengths,
        int* emitted_tokens, float* emitted_timestamps, int* emitted_lengths, int num_frames,
        int blank_id, float encoder_frame_shift_sec)
    {
        // One thread owns an utterance. CTC collapse is inherently sequential,
        // and batching supplies enough independent blocks for GPU parallelism.
        const int utterance = blockIdx.x;
        int previous_token = blank_id;
        int emitted_count = 0;
        const int requested_length = output_lengths[utterance];
        const int length = requested_length < 0
                               ? 0
                               : (requested_length < num_frames ? requested_length : num_frames);
        // Widen only flattened addressing; frame counters stay 32-bit for the
        // common path while large batch-by-frame products remain well-defined.
        const long long utterance_base = static_cast<long long>(utterance) * num_frames;

        for (int frame_index = 0; frame_index < length; ++frame_index)
        {
            const int token = paths[utterance_base + frame_index];
            if (token != previous_token && token != blank_id)
            {
                const long long output_index = utterance_base + emitted_count;
                emitted_tokens[output_index] = token;
                emitted_timestamps[output_index] = frame_index * encoder_frame_shift_sec;
                ++emitted_count;
            }
            previous_token = token;
        }
        emitted_lengths[utterance] = emitted_count;
    }
    """,
    "ctc_collapse",
    options=("--std=c++20",),
    backend="nvcc",
)


HISTORY_HELPERS_SOURCE = r"""
    // Called only after candidate lengths and fingerprints match. Nodes belong
    // to append-only, acyclic histories. A shared node proves the remaining
    // prefixes equal, even when their timestamps or recurrent states differ.
    // Optional memoization has a power-of-two capacity and must be cleared
    // whenever node IDs are recycled for another decoder call.
    __device__ __forceinline__ bool histories_equal(int left, int right, const int* node_parents,
        const int* node_tokens, unsigned long long* history_cache = nullptr, int cache_size = 0)
    {
        unsigned long long first_pair = 0;
        int first_slot = 0;
        while (left != right)
        {
            if (left < 0 || right < 0)
            {
                return false;
            }
            if (history_cache != nullptr)
            {
                // Pack the exact ordered pair of nonnegative int32 node IDs,
                // not a fingerprint. Atomic entries make concurrent eviction
                // harmless: a different pair is always a miss, never a match.
                const unsigned int lower = left < right ? left : right;
                const unsigned int upper = left < right ? right : left;
                const unsigned long long pair =
                    (static_cast<unsigned long long>(lower) << 32) | upper;
                unsigned long long mixed = (pair ^ (pair >> 32)) * 1099511628211ULL;
                mixed ^= mixed >> 32;
                const int slot = static_cast<unsigned int>(mixed) & (cache_size - 1);
                if (first_pair == 0)
                {
                    first_pair = pair;
                    first_slot = slot;
                }
                if (atomicCAS(history_cache + slot, 0ULL, 0ULL) == pair)
                {
                    if (pair == first_pair)
                    {
                        return true;
                    }
                    break;
                }
            }
            if (node_tokens[left] != node_tokens[right])
            {
                return false;
            }
            left = node_parents[left];
            right = node_parents[right];
        }
        if (first_pair != 0)
        {
            atomicExch(history_cache + first_slot, first_pair);
        }
        return true;
    }
"""


ZIPFORMER_BEAM_SEARCH_SOURCE = (
    HISTORY_HELPERS_SOURCE
    + r"""
    #include <climits>
    #include <cstdint>
    #include <cuda_bf16.h>
    #include <cuda_fp16.h>
    #include <math_constants.h>

    #ifndef ZIPFORMER_BEAM
    #error "ZIPFORMER_BEAM must be defined"
    #endif
    #ifndef ZIPFORMER_VOCAB_SIZE
    #error "ZIPFORMER_VOCAB_SIZE must be defined"
    #endif
    #ifndef ZIPFORMER_CONTEXT_SIZE
    #error "ZIPFORMER_CONTEXT_SIZE must be defined"
    #endif
    #ifndef ZIPFORMER_BEAM_SEARCH_THREADS
    #error "ZIPFORMER_BEAM_SEARCH_THREADS must be defined"
    #endif
    #ifndef ZIPFORMER_REGISTER_TOPK
    #define ZIPFORMER_REGISTER_TOPK 0
    #endif

    static_assert(sizeof(int) == 4);
    static_assert(ZIPFORMER_BEAM > 0);
    static_assert(ZIPFORMER_VOCAB_SIZE > 0);
    static_assert(ZIPFORMER_VOCAB_SIZE < INT_MAX);
    static_assert(ZIPFORMER_CONTEXT_SIZE > 0);
    static_assert(ZIPFORMER_BEAM_SEARCH_THREADS > 0);
    static_assert(ZIPFORMER_BEAM_SEARCH_THREADS <= 1024);
    static_assert(ZIPFORMER_BEAM_SEARCH_THREADS % 32 == 0);
    static_assert(ZIPFORMER_REGISTER_TOPK == 0 || ZIPFORMER_REGISTER_TOPK == 1);
    static_assert(ZIPFORMER_BEAM <= INT_MAX / ZIPFORMER_VOCAB_SIZE);
    static_assert(ZIPFORMER_BEAM <= INT_MAX / ZIPFORMER_CONTEXT_SIZE);

    constexpr int ZIPFORMER_FLOAT32 = 0;
    constexpr int ZIPFORMER_FLOAT16 = 1;
    constexpr int ZIPFORMER_BFLOAT16 = 2;

    __device__ __forceinline__ float zipformer_lowest_score() { return -CUDART_INF_F; }

    __device__ __forceinline__ float zipformer_removed_score() { return CUDART_NAN_F; }

    __device__ __forceinline__ float load_zipformer_value(const void* values, int index, int dtype)
    {
        if (dtype == ZIPFORMER_FLOAT16)
        {
            return __half2float(reinterpret_cast<const half*>(values)[index]);
        }
        if (dtype == ZIPFORMER_BFLOAT16)
        {
            return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(values)[index]);
        }
        if (dtype == ZIPFORMER_FLOAT32)
        {
            return reinterpret_cast<const float*>(values)[index];
        }
        return 0.0F;
    }

    __device__ __forceinline__ void store_zipformer_value(
        void* values, int index, int dtype, float value)
    {
        if (dtype == ZIPFORMER_FLOAT16)
        {
            reinterpret_cast<half*>(values)[index] = __float2half(value);
        }
        else if (dtype == ZIPFORMER_BFLOAT16)
        {
            reinterpret_cast<__nv_bfloat16*>(values)[index] = __float2bfloat16(value);
        }
        else if (dtype == ZIPFORMER_FLOAT32)
        {
            reinterpret_cast<float*>(values)[index] = value;
        }
    }

    __device__ __forceinline__ bool zipformer_score_is_better(
        float score, int index, float other_score, int other_index)
    {
        return score > other_score || (score == other_score && index < other_index);
    }

    __device__ __forceinline__ float zipformer_candidate_score(const float* log_probs,
        const float* hypothesis_scores, int parent, int token, int blank_id, float blank_penalty)
    {
        float score = hypothesis_scores[parent] + log_probs[parent * ZIPFORMER_VOCAB_SIZE + token];
        if (token == blank_id)
        {
            score -= blank_penalty;
        }
        // Positive infinity must not outrank finite paths and then terminate
        // materialization of the selected beam below.
        return isfinite(score) ? score : zipformer_lowest_score();
    }

    __device__ __forceinline__ void zipformer_warp_best(float& score, int& index)
    {
        // Every launched block contains complete warps and no lane exits before
        // this helper. All lanes therefore participate safely; only lane zero's
        // final pair is consumed by the block-wide reduction.
        for (int offset = 16; offset > 0; offset /= 2)
        {
            const float other_score = __shfl_down_sync(0xffffffff, score, offset);
            const int other_index = __shfl_down_sync(0xffffffff, index, offset);
            if (zipformer_score_is_better(other_score, other_index, score, index))
            {
                score = other_score;
                index = other_index;
            }
        }
    }

    extern "C" __global__ void zipformer_beam_search(const float* log_probs,
        const void* encoder_output_raw, void* encoder_input_raw, const void* context_lookup_raw,
        void* decoder_input_raw, int* contexts, const float* hypothesis_scores,
        const int* hypothesis_nodes, const int* hypothesis_lengths,
        const unsigned long long* hypothesis_hashes, float* next_scores, int* next_nodes,
        int* next_lengths, unsigned long long* next_hashes, int* node_parents, int* node_tokens,
        float* node_timestamps, int* node_counts, const int* output_lengths, int frame_index,
        int max_frames, int encoder_dim, int encoder_output_dtype, int encoder_input_dtype,
        int context_lookup_dtype, int blank_id, float blank_penalty, float encoder_frame_shift_sec)
    {
        const int beam = ZIPFORMER_BEAM;
        const int vocab_size = ZIPFORMER_VOCAB_SIZE;
        const int context_size = ZIPFORMER_CONTEXT_SIZE;
        const int context_lookup_values = vocab_size + 1;
        const int utterance = blockIdx.x;
        const int thread = threadIdx.x;
        const int hypothesis_base = utterance * beam;
        const long long context_base = static_cast<long long>(hypothesis_base) * context_size;
        const int node_capacity = beam * max_frames;
        const int node_base = utterance * node_capacity;

        // Lengths stay on the GPU. Clamp them here to avoid a synchronization
        // solely for validation and to prevent an invalid next-frame read.
        const int requested_length = output_lengths[utterance];
        const int output_length =
            requested_length < 0 ? 0
                                 : (requested_length < max_frames ? requested_length : max_frames);

        if (frame_index >= output_length)
        {
            // Search state uses ping-pong buffers. Finished utterances still copy
            // their compact state so every utterance has the same final parity.
            for (int index = thread; index < beam; index += blockDim.x)
            {
                next_scores[hypothesis_base + index] = hypothesis_scores[hypothesis_base + index];
                next_nodes[hypothesis_base + index] = hypothesis_nodes[hypothesis_base + index];
                next_lengths[hypothesis_base + index] = hypothesis_lengths[hypothesis_base + index];
                next_hashes[hypothesis_base + index] = hypothesis_hashes[hypothesis_base + index];
            }
            return;
        }

        const int candidate_count = beam * vocab_size;
        // Dynamic shared memory contains, in order, optional candidate scores,
        // one score/index pair per warp, the selected score/index pairs, a
        // snapshot of every parent predictor context, and sparse merge pairs.
        // All regions are 4-byte aligned; the Python launch helper computes
        // this exact byte count.
        extern __shared__ unsigned char shared_memory[];
        const int lane = thread & 31;
        const int warp = thread >> 5;
        const int num_warps = blockDim.x >> 5;
    #if ZIPFORMER_REGISTER_TOPK
        float* reduction_scores = reinterpret_cast<float*>(shared_memory);
    #else
        float* candidate_scores = reinterpret_cast<float*>(shared_memory);
        float* reduction_scores = candidate_scores + candidate_count;
    #endif
        int* reduction_indexes = reinterpret_cast<int*>(reduction_scores + num_warps);
        float* selected_scores = reinterpret_cast<float*>(reduction_indexes + num_warps);
        int* selected_indexes = reinterpret_cast<int*>(selected_scores + beam);
        int* current_contexts = selected_indexes + beam;
        int* merged_indexes = current_contexts + beam * context_size;
        int* removed_indexes = merged_indexes + beam;
        float* merged_scores = reinterpret_cast<float*>(removed_indexes + beam);

        // Contexts are updated in place below. Preserve the parent rows before
        // thread zero starts writing the next generation.
        for (int index = thread; index < beam * context_size; index += blockDim.x)
        {
            current_contexts[index] = contexts[context_base + index];
        }

        // Live parent histories are unique. Nonblank extensions therefore only
        // duplicate a longer parent's blank path. Find these pairs in O(beam^2)
        // rather than hashing the entire hypothesis-by-vocabulary table.
        for (int parent = thread; parent < beam; parent += blockDim.x)
        {
            merged_indexes[parent] = -1;
            const int parent_index = hypothesis_base + parent;
            const int node = hypothesis_nodes[parent_index];
            if (node < 0 || !isfinite(hypothesis_scores[parent_index]))
            {
                continue;
            }
            const int token = node_tokens[node];
            for (int shorter = 0; shorter < beam; ++shorter)
            {
                const int shorter_index = hypothesis_base + shorter;
                // Unsigned wrap is intentional: fingerprints filter candidates,
                // but only matching token histories may merge.
                if (hypothesis_lengths[shorter_index] + 1 != hypothesis_lengths[parent_index]
                    || hypothesis_hashes[shorter_index] * 1099511628211ULL
                               + static_cast<unsigned long long>(token + 1)
                           != hypothesis_hashes[parent_index])
                {
                    continue;
                }
                // The longer history already ends in token, so compare its
                // prefix with the shorter history before merging the extension.
                if (!histories_equal(hypothesis_nodes[shorter_index], node_parents[node],
                        node_parents, node_tokens))
                {
                    continue;
                }
                const float blank_score = zipformer_candidate_score(
                    log_probs, hypothesis_scores, parent_index, blank_id, blank_id, blank_penalty);
                const float token_score = zipformer_candidate_score(
                    log_probs, hypothesis_scores, shorter_index, token, blank_id, blank_penalty);
                if (isfinite(blank_score) && isfinite(token_score))
                {
                    const int blank_candidate = parent * vocab_size + blank_id;
                    const int token_candidate = shorter * vocab_size + token;
                    const bool keep_blank = zipformer_score_is_better(
                        blank_score, blank_candidate, token_score, token_candidate);
                    merged_indexes[parent] = keep_blank ? blank_candidate : token_candidate;
                    removed_indexes[parent] = keep_blank ? token_candidate : blank_candidate;
                    const float maximum = fmaxf(blank_score, token_score);
                    merged_scores[parent] =
                        maximum + logf(expf(blank_score - maximum) + expf(token_score - maximum));
                }
                break;
            }
        }
        __syncthreads();

    #if ZIPFORMER_REGISTER_TOPK
        constexpr int items_per_thread =
            (ZIPFORMER_BEAM * ZIPFORMER_VOCAB_SIZE - 1) / ZIPFORMER_BEAM_SEARCH_THREADS + 1;
        // Sort short per-thread lists instead of rescanning shared scores for
        // each rank. Dynamic indexing may place these lists in local memory;
        // the register variant's name does not guarantee spill-free code.
        float local_scores[items_per_thread];
        int local_indexes[items_per_thread];
    #pragma unroll
        for (int item = 0; item < items_per_thread; ++item)
        {
            const int candidate = thread + item * ZIPFORMER_BEAM_SEARCH_THREADS;
            float score = zipformer_lowest_score();
            int candidate_index = candidate_count;
            if (candidate < candidate_count)
            {
                const int parent = candidate / vocab_size;
                const int token = candidate - parent * vocab_size;
                score = zipformer_candidate_score(log_probs, hypothesis_scores,
                    hypothesis_base + parent, token, blank_id, blank_penalty);
                candidate_index = candidate;
            }

            local_scores[item] = score;
            local_indexes[item] = candidate_index;
        }
        for (int parent = 0; parent < beam; ++parent)
        {
            const int winner = merged_indexes[parent];
            if (winner >= 0)
            {
                const int loser = removed_indexes[parent];
                if (winner % ZIPFORMER_BEAM_SEARCH_THREADS == thread)
                {
                    local_scores[winner / ZIPFORMER_BEAM_SEARCH_THREADS] = merged_scores[parent];
                }
                if (loser % ZIPFORMER_BEAM_SEARCH_THREADS == thread)
                {
                    local_scores[loser / ZIPFORMER_BEAM_SEARCH_THREADS] = zipformer_lowest_score();
                }
            }
        }

    #pragma unroll
        for (int item = 1; item < items_per_thread; ++item)
        {
            const float score = local_scores[item];
            const int candidate_index = local_indexes[item];
            int position = item;
    #pragma unroll
            while (position > 0
                   && zipformer_score_is_better(score, candidate_index, local_scores[position - 1],
                       local_indexes[position - 1]))
            {
                local_scores[position] = local_scores[position - 1];
                local_indexes[position] = local_indexes[position - 1];
                --position;
            }
            local_scores[position] = score;
            local_indexes[position] = candidate_index;
        }
        __syncthreads();

        int local_position = 0;
    #pragma unroll
        for (int rank = 0; rank < beam; ++rank)
        {
            float best_score = local_position < items_per_thread ? local_scores[local_position]
                                                                 : zipformer_lowest_score();
            int best_index =
                local_position < items_per_thread ? local_indexes[local_position] : candidate_count;
            zipformer_warp_best(best_score, best_index);
            if (lane == 0)
            {
                reduction_scores[warp] = best_score;
                reduction_indexes[warp] = best_index;
            }
            __syncthreads();

            if (warp == 0)
            {
                best_score = lane < num_warps ? reduction_scores[lane] : zipformer_lowest_score();
                best_index = lane < num_warps ? reduction_indexes[lane] : candidate_count;
                zipformer_warp_best(best_score, best_index);
                if (lane == 0)
                {
                    selected_scores[rank] = best_score;
                    selected_indexes[rank] = best_index;
                }
            }
            __syncthreads();
            if (local_position < items_per_thread
                && local_indexes[local_position] == selected_indexes[rank])
            {
                ++local_position;
            }
        }
    #else
        // The occupancy-oriented variant materializes all candidate scores once,
        // then removes one block-wide maximum for each retained rank.
        for (int candidate = thread; candidate < candidate_count; candidate += blockDim.x)
        {
            const int parent = candidate / vocab_size;
            const int token = candidate - parent * vocab_size;
            candidate_scores[candidate] = zipformer_candidate_score(log_probs, hypothesis_scores,
                hypothesis_base + parent, token, blank_id, blank_penalty);
        }
        __syncthreads();
        for (int parent = thread; parent < beam; parent += blockDim.x)
        {
            const int winner = merged_indexes[parent];
            if (winner >= 0)
            {
                candidate_scores[winner] = merged_scores[parent];
                candidate_scores[removed_indexes[parent]] = zipformer_lowest_score();
            }
        }
        __syncthreads();

        for (int rank = 0; rank < beam; ++rank)
        {
            float best_score = zipformer_lowest_score();
            int best_index = candidate_count;
            for (int candidate = thread; candidate < candidate_count; candidate += blockDim.x)
            {
                const float score = candidate_scores[candidate];
                if (zipformer_score_is_better(score, candidate, best_score, best_index))
                {
                    best_score = score;
                    best_index = candidate;
                }
            }
            zipformer_warp_best(best_score, best_index);
            if (lane == 0)
            {
                reduction_scores[warp] = best_score;
                reduction_indexes[warp] = best_index;
            }
            __syncthreads();

            if (warp == 0)
            {
                best_score = lane < num_warps ? reduction_scores[lane] : zipformer_lowest_score();
                best_index = lane < num_warps ? reduction_indexes[lane] : candidate_count;
                zipformer_warp_best(best_score, best_index);
                if (lane == 0)
                {
                    selected_scores[rank] = best_score;
                    selected_indexes[rank] = best_index;
                    // NaN never wins the ordered comparison, unlike -infinity,
                    // which remains a valid log probability for another candidate.
                    candidate_scores[best_index] = zipformer_removed_score();
                }
            }
            __syncthreads();
        }
    #endif

        if (thread == 0)
        {
            for (int index = 0; index < beam; ++index)
            {
                next_scores[hypothesis_base + index] = zipformer_lowest_score();
                next_nodes[hypothesis_base + index] = -1;
                next_lengths[hypothesis_base + index] = 0;
                next_hashes[hypothesis_base + index] = 0;
                const long long output_context_base =
                    static_cast<long long>(hypothesis_base + index) * context_size;
                for (int context = 0; context < context_size; ++context)
                {
                    contexts[output_context_base + context] = 0;
                }
            }

            for (int rank = 0; rank < beam; ++rank)
            {
                if (!isfinite(selected_scores[rank]))
                {
                    break;
                }
                const int selected_index = selected_indexes[rank];
                const int parent = selected_index / vocab_size;
                const int token = selected_index - parent * vocab_size;
                const bool emitted = token != blank_id;
                const int parent_index = hypothesis_base + parent;
                const int parent_node = hypothesis_nodes[parent_index];
                const int candidate_length = hypothesis_lengths[parent_index] + emitted;
                // Unsigned wrap is intentional: this is a modulo-2^64 history
                // fingerprint used only for equality checks, never addressing.
                const unsigned long long candidate_hash =
                    emitted ? hypothesis_hashes[parent_index] * 1099511628211ULL
                                  + static_cast<unsigned long long>(token + 1)
                            : hypothesis_hashes[parent_index];

                int candidate_node = parent_node;
                if (emitted)
                {
                    // At most `beam` candidates emit per frame, so the per-
                    // utterance allocation of `beam * max_frames` nodes is exact.
                    candidate_node = node_base + node_counts[utterance];
                    ++node_counts[utterance];
                    node_parents[candidate_node] = parent_node;
                    node_tokens[candidate_node] = token;
                    node_timestamps[candidate_node] =
                        roundf(frame_index * encoder_frame_shift_sec * 1000.0F) / 1000.0F;
                }

                const int output_index = hypothesis_base + rank;
                const long long output_context_base =
                    static_cast<long long>(output_index) * context_size;
                next_scores[output_index] = selected_scores[rank];
                next_nodes[output_index] = candidate_node;
                next_lengths[output_index] = candidate_length;
                next_hashes[output_index] = candidate_hash;
                if (emitted)
                {
                    for (int context = 0; context < context_size - 1; ++context)
                    {
                        contexts[output_context_base + context] =
                            current_contexts[parent * context_size + context + 1];
                    }
                    contexts[output_context_base + context_size - 1] = token;
                }
                else
                {
                    for (int context = 0; context < context_size; ++context)
                    {
                        contexts[output_context_base + context] =
                            current_contexts[parent * context_size + context];
                    }
                }
            }

            for (int hypothesis = 0; hypothesis < beam; ++hypothesis)
            {
                int lookup_index = 0;
                const long long context_index =
                    static_cast<long long>(hypothesis_base + hypothesis) * context_size;
                for (int context = 0; context < context_size; ++context)
                {
                    lookup_index = lookup_index * context_lookup_values
                                   + contexts[context_index + context] + 1;
                }
                selected_indexes[hypothesis] = lookup_index;
            }
        }
        __syncthreads();

        const int next_frame = frame_index + 1;
        if (next_frame < output_length)
        {
            const int encoder_output_base = (utterance * max_frames + next_frame) * encoder_dim;
            const int encoder_input_base = hypothesis_base * encoder_dim;
            const bool same_dtype = encoder_output_dtype == encoder_input_dtype
                                    && encoder_output_dtype == context_lookup_dtype;
            const int packed_elements = encoder_output_dtype == ZIPFORMER_FLOAT32 ? 4 : 8;
            const uintptr_t addresses = reinterpret_cast<uintptr_t>(encoder_output_raw)
                                        | reinterpret_cast<uintptr_t>(encoder_input_raw)
                                        | reinterpret_cast<uintptr_t>(context_lookup_raw)
                                        | reinterpret_cast<uintptr_t>(decoder_input_raw);
            const bool can_copy_16_bytes = same_dtype
                                           && (encoder_output_dtype == ZIPFORMER_FLOAT32
                                               || encoder_output_dtype == ZIPFORMER_FLOAT16
                                               || encoder_output_dtype == ZIPFORMER_BFLOAT16)
                                           && encoder_dim % packed_elements == 0
                                           && addresses % alignof(uint4) == 0;
            if (can_copy_16_bytes)
            {
                // Check both base addresses and row widths. Contiguous slices
                // need not start at a 16-byte boundary. Aligned values need no
                // conversion, so uint4 handles all three dtypes bit-for-bit.
                const int packed_dim = encoder_dim / packed_elements;
                const uint4* encoder_output_packed =
                    reinterpret_cast<const uint4*>(encoder_output_raw);
                uint4* encoder_input_packed = reinterpret_cast<uint4*>(encoder_input_raw);
                const uint4* context_lookup_packed =
                    reinterpret_cast<const uint4*>(context_lookup_raw);
                uint4* decoder_input_packed = reinterpret_cast<uint4*>(decoder_input_raw);
                const int encoder_output_packed_base =
                    (utterance * max_frames + next_frame) * packed_dim;
                const int hypothesis_packed_base = hypothesis_base * packed_dim;
                for (unsigned int index = thread; index < beam * packed_dim; index += blockDim.x)
                {
                    const int hypothesis = index / packed_dim;
                    const int feature = index - hypothesis * packed_dim;
                    encoder_input_packed[hypothesis_packed_base + index] =
                        encoder_output_packed[encoder_output_packed_base + feature];
                    decoder_input_packed[hypothesis_packed_base + index] =
                        context_lookup_packed[selected_indexes[hypothesis] * packed_dim + feature];
                }
            }
            else
            {
                // The final stride may exceed INT_MAX even when every accessed
                // element fits. An unsigned counter exits instead of wrapping negative.
                for (unsigned int index = thread; index < beam * encoder_dim; index += blockDim.x)
                {
                    const int hypothesis = index / encoder_dim;
                    const int feature = index - hypothesis * encoder_dim;
                    const int output_index = encoder_output_base + feature;
                    const float value = load_zipformer_value(
                        encoder_output_raw, output_index, encoder_output_dtype);
                    const int context_index = selected_indexes[hypothesis] * encoder_dim + feature;
                    const float context_value = load_zipformer_value(
                        context_lookup_raw, context_index, context_lookup_dtype);
                    store_zipformer_value(
                        encoder_input_raw, encoder_input_base + index, encoder_input_dtype, value);
                    store_zipformer_value(decoder_input_raw, encoder_input_base + index,
                        encoder_input_dtype, context_value);
                }
            }
        }
    }
    """
)


@cache
def get_zipformer_beam_search_kernels(
    beam: int, vocab_size: int, context_size: int
) -> tuple[tuple[cp.RawKernel, int] | None, tuple[cp.RawKernel, int], int]:
    """Return optional register and mandatory shared launch configurations.

    Parameters
    ----------
    beam : int
        Number of hypotheses retained for each utterance.
    vocab_size : int
        Number of token candidates produced for each hypothesis.
    context_size : int
        Number of token IDs forming one predictor context.

    Returns
    -------
    register_launch : tuple[cp.RawKernel, int] or None
        Register-local top-k kernel and its required dynamic shared memory in
        bytes. ``None`` indicates that the search shape would require excessive
        per-thread register storage.
    shared_launch : tuple[cp.RawKernel, int]
        Shared-memory top-k kernel and its required dynamic shared memory in
        bytes. This variant avoids per-thread candidate arrays and is the
        occupancy-oriented fallback for larger batches.
    threads : int
        Number of CUDA threads required by either kernel variant.

    Raises
    ------
    ValueError
        Raised when the configured thread block is incompatible with the
        block-wide reductions or the dynamic shared-memory requirement exceeds
        the signed 32-bit launch-size range.

    Notes
    -----
    Both variants compile ``beam``, ``vocab_size``, and ``context_size`` into the
    CUDA source. Callers must supply validated positive integer shapes.
    The register-local variant is emitted only for warp-aligned
    blocks, beams no larger than eight, and at most eight candidates per thread.
    It minimizes synchronization for small launches, while the shared-memory
    variant retains occupancy for large batches. Results are cached by search
    shape so each configuration creates its CuPy kernels only once.
    The compiler can place dynamically indexed candidate lists in local memory;
    compare compiled resource usage and timings before changing the dispatch rule.
    """

    threads = ZIPFORMER_BEAM_SEARCH_THREADS
    if not isinstance(threads, int) or not 0 < threads <= 1024 or threads % 32 != 0:
        raise ValueError(
            "ZIPFORMER_BEAM_SEARCH_THREADS must be a positive multiple of 32 "
            f"no larger than 1024, got {threads}."
        )

    candidates_per_thread = (beam * vocab_size - 1) // threads + 1
    register_topk_supported = beam <= 8 and candidates_per_thread <= 8

    reduction_bytes = (
        threads // 32 * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
    )
    register_shared_memory_bytes = reduction_bytes + beam * (
        2 * np.dtype(np.float32).itemsize
        + 3 * np.dtype(np.int32).itemsize
        + context_size * np.dtype(np.int32).itemsize
    )
    shared_memory_bytes = (
        register_shared_memory_bytes + beam * vocab_size * np.dtype(np.float32).itemsize
    )
    if shared_memory_bytes > INT32_MAX:
        raise ValueError(
            "Zipformer beam-search dynamic shared memory exceeds the signed "
            f"32-bit launch-size limit: {shared_memory_bytes} bytes."
        )

    common_options = (
        "--std=c++20",
        f"-DZIPFORMER_BEAM={beam}",
        f"-DZIPFORMER_VOCAB_SIZE={vocab_size}",
        f"-DZIPFORMER_CONTEXT_SIZE={context_size}",
        f"-DZIPFORMER_BEAM_SEARCH_THREADS={threads}",
    )
    shared_kernel = cp.RawKernel(
        ZIPFORMER_BEAM_SEARCH_SOURCE,
        "zipformer_beam_search",
        options=(*common_options, "-DZIPFORMER_REGISTER_TOPK=0"),
        backend="nvcc",
    )
    if not register_topk_supported:
        return None, (shared_kernel, shared_memory_bytes), threads

    register_kernel = cp.RawKernel(
        ZIPFORMER_BEAM_SEARCH_SOURCE,
        "zipformer_beam_search",
        options=(*common_options, "-DZIPFORMER_REGISTER_TOPK=1"),
        backend="nvcc",
    )
    return (
        (register_kernel, register_shared_memory_bytes),
        (shared_kernel, shared_memory_bytes),
        threads,
    )


ZIPFORMER_FINALIZE_KERNEL = cp.RawKernel(
    r"""
    extern "C" __global__ void zipformer_finalize(const float* hypothesis_scores,
        const int* hypothesis_nodes, const int* hypothesis_lengths, const int* node_parents,
        const int* node_tokens, const float* node_timestamps, int* output_tokens,
        float* output_timestamps, int* output_lengths, int max_frames, int beam, int context_size)
    {
        const int utterance = blockIdx.x;
        const int hypothesis_base = utterance * beam;
        int best_hypothesis = hypothesis_base;
        int best_length = hypothesis_lengths[best_hypothesis];
        // Include the fixed predictor context in the normalization denominator,
        // matching Icefall's modified beam-search final ranking.
        double best_score = static_cast<double>(hypothesis_scores[best_hypothesis])
                            / (static_cast<double>(best_length) + context_size);
        for (int hypothesis = 1; hypothesis < beam; ++hypothesis)
        {
            const int index = hypothesis_base + hypothesis;
            const int length = hypothesis_lengths[index];
            const double score = static_cast<double>(hypothesis_scores[index])
                                 / (static_cast<double>(length) + context_size);
            if (score > best_score)
            {
                best_hypothesis = index;
                best_length = length;
                best_score = score;
            }
        }

        output_lengths[utterance] = best_length;
        int node = hypothesis_nodes[best_hypothesis];
        // Histories are stored newest-first as parent links. Reconstructing from
        // the last output position restores chronological token order.
        for (int position = best_length - 1; position >= 0; --position)
        {
            const int output_index = utterance * max_frames + position;
            output_tokens[output_index] = node_tokens[node];
            output_timestamps[output_index] = node_timestamps[node];
            node = node_parents[node];
        }
    }
    """,
    "zipformer_finalize",
    options=("--std=c++20",),
    backend="nvcc",
)


TDT_VALUE_HELPERS_SOURCE = r"""
    #include <cstdint>
    #include <cuda_bf16.h>
    #include <cuda_fp16.h>

    // TensorRT buffers arrive as void pointers, so kernels carry one explicit
    // dtype tag matching the runtime decoder's kernel-dtype map.
    constexpr int TDT_FLOAT32 = 0;
    constexpr int TDT_FLOAT16 = 1;
    constexpr int TDT_BFLOAT16 = 2;

    __device__ __forceinline__ float load_tdt_value(const void* values, int index, int dtype)
    {
        if (dtype == TDT_FLOAT16)
        {
            return __half2float(reinterpret_cast<const half*>(values)[index]);
        }
        if (dtype == TDT_BFLOAT16)
        {
            return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(values)[index]);
        }
        return reinterpret_cast<const float*>(values)[index];
    }

    __device__ __forceinline__ void store_tdt_value(void* values, int index, int dtype, float value)
    {
        if (dtype == TDT_FLOAT16)
        {
            reinterpret_cast<half*>(values)[index] = __float2half(value);
        }
        else if (dtype == TDT_BFLOAT16)
        {
            reinterpret_cast<__nv_bfloat16*>(values)[index] = __float2bfloat16(value);
        }
        else
        {
            reinterpret_cast<float*>(values)[index] = value;
        }
    }

    __device__ __forceinline__ int tdt_clamp_output_length(int requested_length, int num_frames)
    {
        return requested_length < 0
                   ? 0
                   : (requested_length < num_frames ? requested_length : num_frames);
    }

    __device__ __forceinline__ void stage_tdt_encoder_input(const void* encoder_output_raw,
        void* encoder_input_raw, int output_base, int input_base, int encoder_dim,
        int encoder_output_dtype, int encoder_input_dtype, bool active, int thread)
    {
        const bool same_dtype = encoder_output_dtype == encoder_input_dtype;
        const int packed_elements = encoder_output_dtype == TDT_FLOAT32 ? 4 : 8;
        const uintptr_t addresses = reinterpret_cast<uintptr_t>(encoder_output_raw)
                                    | reinterpret_cast<uintptr_t>(encoder_input_raw);
        const bool can_copy_16_bytes =
            same_dtype && encoder_dim % packed_elements == 0 && addresses % alignof(uint4) == 0;
        if (can_copy_16_bytes)
        {
            // Both base addresses and row widths are aligned; contiguous views
            // with an offset instead take the scalar path below.
            const int encoder_vectors = encoder_dim / packed_elements;
            const uint4* encoder_output = reinterpret_cast<const uint4*>(encoder_output_raw);
            uint4* encoder_input = reinterpret_cast<uint4*>(encoder_input_raw);
            const int output_vector_base = output_base / packed_elements;
            const int input_vector_base = input_base / packed_elements;
            for (unsigned int feature = thread; feature < encoder_vectors; feature += blockDim.x)
            {
                encoder_input[input_vector_base + feature] =
                    active ? encoder_output[output_vector_base + feature] : make_uint4(0, 0, 0, 0);
            }
            return;
        }

        // The final stride may pass INT_MAX; only in-bounds offsets are converted
        // back to int. The scalar path also handles mixed encoder precisions.
        for (unsigned int feature = thread; feature < encoder_dim; feature += blockDim.x)
        {
            const float value = active ? load_tdt_value(encoder_output_raw, output_base + feature,
                                             encoder_output_dtype)
                                       : 0.0F;
            store_tdt_value(encoder_input_raw, input_base + feature, encoder_input_dtype, value);
        }
    }
    """

TDT_TOPK_HELPERS_SOURCE = r"""
    #include <math_constants.h>

    __device__ __forceinline__ float tdt_lowest_score() { return -CUDART_INF_F; }

    __device__ __forceinline__ float tdt_removed_score() { return CUDART_NAN_F; }

    __device__ __forceinline__ bool tdt_score_is_better(
        float score, int index, float other_score, int other_index)
    {
        return score > other_score || (score == other_score && index < other_index);
    }

    __device__ __forceinline__ void tdt_warp_best(float& score, int& index)
    {
        // Every caller launches complete warps and keeps all lanes active through
        // the reduction, so the full-warp mask is valid here.
        for (int offset = 16; offset > 0; offset /= 2)
        {
            const float other_score = __shfl_down_sync(0xffffffff, score, offset);
            const int other_index = __shfl_down_sync(0xffffffff, index, offset);
            if (tdt_score_is_better(other_score, other_index, score, index))
            {
                score = other_score;
                index = other_index;
            }
        }
    }
    """

TDT_SCORE_HELPERS_SOURCE = r"""
    __device__ __forceinline__ float tdt_merge_log_scores(float first, float second)
    {
        // Stable two-term logaddexp. Preserve infinities explicitly because
        // subtracting -infinity from itself would otherwise produce NaN.
        const float maximum = fmaxf(first, second);
        return isinf(maximum) ? maximum
                              : maximum + logf(expf(first - maximum) + expf(second - maximum));
    }

    __device__ __forceinline__ double tdt_length_normalized_score(float score, int length)
    {
        return static_cast<double>(score) / (length > 0 ? length : 1);
    }

    __device__ __forceinline__ int tdt_advance_time(
        int time_index, int duration, bool force_advance, int output_length)
    {
        // Use a wide intermediate before clamping to the valid encoder range.
        const long long advanced_time =
            static_cast<long long>(time_index) + duration + static_cast<int>(force_advance);
        return advanced_time < output_length ? static_cast<int>(advanced_time) : output_length;
    }

    __device__ __forceinline__ void tdt_advance_search_state(bool emitted, int duration,
        int time_index, int symbols_at_timestep, int max_symbols_per_timestep, int output_length,
        int& next_time_index, int& next_symbols_at_timestep)
    {
        next_symbols_at_timestep = 0;
        bool force_advance = false;
        if (emitted && duration == 0)
        {
            next_symbols_at_timestep = symbols_at_timestep + 1;
            if (next_symbols_at_timestep >= max_symbols_per_timestep)
            {
                force_advance = true;
                next_symbols_at_timestep = 0;
            }
        }
        next_time_index = tdt_advance_time(time_index, duration, force_advance, output_length);
    }
    """

TDT_PREPARE_INPUTS_KERNEL = cp.RawKernel(
    TDT_VALUE_HELPERS_SOURCE
    + r"""
    extern "C" __global__ void tdt_prepare_inputs(const void* encoder_output_raw,
        const int* output_lengths, const float* hypothesis_scores, const int* time_indexes,
        const int* last_tokens, void* encoder_input_raw, int* targets, int actual_batch_size,
        int num_frames, int encoder_dim, int beam, int encoder_output_dtype,
        int encoder_input_dtype)
    {
        // One block prepares one hypothesis from the fixed engine capacity.
        // Padded or completed hypotheses receive neutral predictor inputs.
        const int hypothesis = blockIdx.x;
        const int utterance = hypothesis / beam;
        const int thread = threadIdx.x;
        const int output_length =
            utterance < actual_batch_size
                ? tdt_clamp_output_length(output_lengths[utterance], num_frames)
                : 0;
        const bool active =
            isfinite(hypothesis_scores[hypothesis]) && time_indexes[hypothesis] < output_length;

        if (thread == 0)
        {
            targets[hypothesis] = active ? last_tokens[hypothesis] : 0;
        }

        const int frame = min(time_indexes[hypothesis], num_frames - 1);
        const int output_base = (utterance * num_frames + frame) * encoder_dim;
        const int input_base = hypothesis * encoder_dim;
        stage_tdt_encoder_input(encoder_output_raw, encoder_input_raw, output_base, input_base,
            encoder_dim, encoder_output_dtype, encoder_input_dtype, active, thread);
    }
    """,
    "tdt_prepare_inputs",
    options=("--std=c++20",),
    backend="nvcc",
)

TDT_SELECT_TOKENS_KERNEL = cp.RawKernel(
    TDT_TOPK_HELPERS_SOURCE
    + r"""
    extern "C" __global__ void tdt_select_tokens(const float* token_log_probs,
        const float* hypothesis_scores, const int* time_indexes, const int* output_lengths,
        float* top_token_scores, int* top_token_indexes, int vocab_size, int beam)
    {
        const int hypothesis = blockIdx.x;
        const int thread = threadIdx.x;
        const int utterance = hypothesis / beam;
        const bool active = isfinite(hypothesis_scores[hypothesis])
                            && time_indexes[hypothesis] < output_lengths[utterance];

        // Blank probability is handled with duration expansion in the beam-search
        // kernel. This stage retains only the best nonblank tokens per parent.
        extern __shared__ unsigned char shared_memory[];
        const int lane = thread & 31;
        const int warp = thread >> 5;
        const int num_warps = blockDim.x >> 5;

        // Shared memory holds all vocabulary scores followed by one reduction
        // score/index pair per warp.
        float* candidate_scores = reinterpret_cast<float*>(shared_memory);
        float* reduction_scores = candidate_scores + vocab_size;
        int* reduction_indexes = reinterpret_cast<int*>(reduction_scores + num_warps);
        for (int token = thread; token < vocab_size; token += blockDim.x)
        {
            float score = active ? token_log_probs[hypothesis * (vocab_size + 1) + token]
                                 : tdt_lowest_score();
            candidate_scores[token] = isfinite(score) ? score : tdt_lowest_score();
        }
        __syncthreads();

        for (int rank = 0; rank < beam; ++rank)
        {
            float best_score = tdt_lowest_score();
            int best_index = vocab_size;
            for (int token = thread; token < vocab_size; token += blockDim.x)
            {
                const float score = candidate_scores[token];
                if (tdt_score_is_better(score, token, best_score, best_index))
                {
                    best_score = score;
                    best_index = token;
                }
            }
            tdt_warp_best(best_score, best_index);
            if (lane == 0)
            {
                reduction_scores[warp] = best_score;
                reduction_indexes[warp] = best_index;
            }
            __syncthreads();

            if (warp == 0)
            {
                best_score = lane < num_warps ? reduction_scores[lane] : tdt_lowest_score();
                best_index = lane < num_warps ? reduction_indexes[lane] : vocab_size;
                tdt_warp_best(best_score, best_index);
                if (lane == 0)
                {
                    top_token_scores[hypothesis * beam + rank] = best_score;
                    top_token_indexes[hypothesis * beam + rank] = best_index;
                    // NaN is a tombstone that cannot win again; -infinity remains
                    // a valid score for another real token in a degenerate row.
                    candidate_scores[best_index] = tdt_removed_score();
                }
            }
            __syncthreads();
        }
    }
    """,
    "tdt_select_tokens",
    options=("--std=c++20",),
    backend="nvcc",
)

TDT_BEAM_SEARCH_KERNEL = cp.RawKernel(
    HISTORY_HELPERS_SOURCE
    + TDT_VALUE_HELPERS_SOURCE
    + TDT_TOPK_HELPERS_SOURCE
    + TDT_SCORE_HELPERS_SOURCE
    + r"""
    struct TdtCandidate
    {
        int parent;
        int duration_index;
        int token;
        int length;
        int time;
        int symbols;
        unsigned long long hash;
    };

    __device__ __forceinline__ TdtCandidate tdt_candidate(int index, int beam, int duration_count,
        int positive_duration_count, int hypothesis_base, int blank_id,
        int max_symbols_per_timestep, int output_length, const int* durations,
        const int* positive_duration_indexes, const int* top_token_indexes,
        const unsigned long long* hypothesis_hashes, const int* hypothesis_lengths,
        const int* time_indexes, const int* symbols_at_timestep)
    {
        TdtCandidate candidate;
        const int per_parent = duration_count * beam;
        const int token_count = per_parent * beam;
        const bool emitted = index < token_count;
        if (emitted)
        {
            candidate.parent = hypothesis_base + index / per_parent;
            candidate.duration_index = index % per_parent / beam;
            candidate.token = top_token_indexes[candidate.parent * beam + index % beam];
        }
        else
        {
            const int blank_index = index - token_count;
            candidate.parent = hypothesis_base + blank_index / positive_duration_count;
            candidate.duration_index =
                positive_duration_indexes[blank_index % positive_duration_count];
            candidate.token = blank_id;
        }
        candidate.length = hypothesis_lengths[candidate.parent] + emitted;
        // The rolling fingerprint is only a filter. Equal keys also require
        // an exact token-history check before their scores can merge.
        candidate.hash = emitted ? hypothesis_hashes[candidate.parent] * 1099511628211ULL
                                       + static_cast<unsigned long long>(candidate.token + 1)
                                 : hypothesis_hashes[candidate.parent];
        tdt_advance_search_state(emitted, durations[candidate.duration_index],
            time_indexes[candidate.parent], symbols_at_timestep[candidate.parent],
            max_symbols_per_timestep, output_length, candidate.time, candidate.symbols);
        if (candidate.time >= output_length)
        {
            candidate.symbols = 0;
        }
        return candidate;
    }

    __device__ __forceinline__ bool tdt_histories_equal(const TdtCandidate& left,
        const TdtCandidate& right, int blank_id, const int* hypothesis_nodes,
        const int* node_parents, const int* node_tokens, unsigned long long* history_cache,
        int cache_size)
    {
        int left_node = hypothesis_nodes[left.parent];
        int right_node = hypothesis_nodes[right.parent];
        // Emitted tokens are not materialized yet. Compare these virtual tails
        // first, removing a stored tail from the blank path when necessary.
        if (left.token != blank_id && right.token != blank_id)
        {
            if (left.token != right.token)
            {
                return false;
            }
        }
        else if (left.token != blank_id)
        {
            if (right_node < 0 || left.token != node_tokens[right_node])
            {
                return false;
            }
            right_node = node_parents[right_node];
        }
        else if (right.token != blank_id)
        {
            if (left_node < 0 || right.token != node_tokens[left_node])
            {
                return false;
            }
            left_node = node_parents[left_node];
        }
        return histories_equal(
            left_node, right_node, node_parents, node_tokens, history_cache, cache_size);
    }

    __device__ __forceinline__ void gather_tdt_states(int hypothesis, int thread,
        const void* input_state_1_raw, const void* input_state_2_raw,
        const void* output_state_1_raw, const void* output_state_2_raw, const int* parent_indexes,
        const unsigned char* use_output_state, const void* encoder_output_raw,
        const float* hypothesis_scores, const int* time_indexes, const int* last_tokens,
        void* next_state_1_raw, void* next_state_2_raw, void* encoder_input_raw, int* targets,
        int output_length, int decoder_capacity, int hidden_dim, int state_layers, int num_frames,
        int encoder_dim, int beam, int state_dtype, int encoder_output_dtype,
        int encoder_input_dtype);

    extern "C" __global__ void tdt_beam_search(const float* token_log_probs,
        const float* duration_log_probs, const float* top_token_scores,
        const int* top_token_indexes, const float* hypothesis_scores, const int* hypothesis_nodes,
        const unsigned long long* hypothesis_hashes, const int* hypothesis_lengths,
        const int* time_indexes, const int* last_tokens, const int* symbols_at_timestep,
        float* next_scores, int* next_nodes, unsigned long long* next_hashes, int* next_lengths,
        int* next_time_indexes, int* next_last_tokens, int* next_symbols_at_timestep,
        int* parent_indexes, unsigned char* use_output_state, int* node_parents, int* node_tokens,
        float* node_timestamps, int* node_counts, float* completed_scores, int* completed_nodes,
        int* completed_lengths, int* active_flags, const int* output_lengths, const int* durations,
        const int* positive_duration_indexes, const void* input_state_1_raw,
        const void* input_state_2_raw, const void* output_state_1_raw,
        const void* output_state_2_raw, void* next_state_1_raw, void* next_state_2_raw,
        const void* encoder_output_raw, void* encoder_input_raw, int* targets, int hidden_dim,
        int state_layers, const int* runtime_dimensions, int encoder_dim, int state_dtype,
        int encoder_output_dtype, int encoder_input_dtype, int token_stride, int beam,
        int duration_count, int positive_duration_count, int blank_id, int max_symbols_per_timestep,
        float blank_penalty, float encoder_frame_shift_sec, unsigned long long* history_cache,
        int history_cache_size)
    {
        // Runtime dimensions live in device memory so a captured search graph
        // can be replayed for different batch and temporal shapes.
        const int actual_batch_size = runtime_dimensions[0];
        const int num_frames = runtime_dimensions[1];
        const int utterance = blockIdx.x;
        const int thread = threadIdx.x;
        // The grid spans the fixed engine batch capacity, not merely the current
        // batch, so this product matches all recurrent-state buffer layouts.
        const int decoder_capacity = static_cast<int>(gridDim.x) * beam;
        const int hypothesis_base = utterance * beam;
        const int output_length =
            utterance < actual_batch_size
                ? tdt_clamp_output_length(output_lengths[utterance], num_frames)
                : 0;
        const int token_candidates_per_parent = duration_count * beam;
        const int token_candidate_count = beam * token_candidates_per_parent;
        const int candidate_count = token_candidate_count + beam * positive_duration_count;
        int bucket_count = 1;
        while (bucket_count < (candidate_count + 1LL) / 2)
        {
            bucket_count *= 2;
        }

        for (int output = thread; output < beam; output += blockDim.x)
        {
            const int output_index = hypothesis_base + output;
            next_scores[output_index] = tdt_lowest_score();
            next_nodes[output_index] = -1;
            next_hashes[output_index] = 0;
            next_lengths[output_index] = 0;
            next_time_indexes[output_index] = output_length;
            next_last_tokens[output_index] = blank_id;
            next_symbols_at_timestep[output_index] = 0;
            parent_indexes[output_index] = -1;
            use_output_state[output_index] = 0;
        }
        if (thread == 0)
        {
            active_flags[utterance] = 0;
        }

        bool has_active_parent = false;
        for (int parent = thread; parent < beam; parent += blockDim.x)
        {
            const int parent_index = hypothesis_base + parent;
            if (isfinite(hypothesis_scores[parent_index])
                && time_indexes[parent_index] < output_length)
            {
                has_active_parent = true;
                break;
            }
        }
        // The block vote detects completion and makes the initialization above
        // visible before the candidate region is reused for top-k reduction.
        if (!__syncthreads_or(has_active_parent))
        {
            return;
        }

        extern __shared__ unsigned char shared_memory[];
        const int lane = thread & 31;
        const int warp = thread >> 5;
        const int num_warps = blockDim.x >> 5;
        // Each candidate belongs to one bucket. Linked lists avoid a full
        // candidate-by-candidate comparison and require no floating-point atomics.
        float* candidate_scores = reinterpret_cast<float*>(shared_memory);
        int* candidate_links = reinterpret_cast<int*>(candidate_scores + candidate_count);
        int* bucket_heads = candidate_links + candidate_count;
        float* reduction_scores = reinterpret_cast<float*>(bucket_heads + bucket_count);
        int* reduction_indexes = reinterpret_cast<int*>(reduction_scores + num_warps);
        float* selected_scores = reinterpret_cast<float*>(reduction_indexes + num_warps);
        int* selected_indexes = reinterpret_cast<int*>(selected_scores + beam);
        for (int bucket = thread; bucket < bucket_count; bucket += blockDim.x)
        {
            bucket_heads[bucket] = -1;
        }
        __syncthreads();

        // Expand each parent with its top nonblank tokens at every duration.
        // Blank candidates use only positive durations so search always makes
        // progress instead of admitting an infinite blank-duration-zero loop.
        for (int candidate = thread; candidate < candidate_count; candidate += blockDim.x)
        {
            const TdtCandidate state = tdt_candidate(candidate, beam, duration_count,
                positive_duration_count, hypothesis_base, blank_id, max_symbols_per_timestep,
                output_length, durations, positive_duration_indexes, top_token_indexes,
                hypothesis_hashes, hypothesis_lengths, time_indexes, symbols_at_timestep);
            const int parent_index = state.parent;
            float score;
            if (candidate < token_candidate_count)
            {
                score = hypothesis_scores[parent_index]
                        + duration_log_probs[parent_index * duration_count + state.duration_index]
                        + top_token_scores[parent_index * beam + candidate % beam];
            }
            else
            {
                score = hypothesis_scores[parent_index]
                        + token_log_probs[parent_index * (blank_id + 1) + blank_id] - blank_penalty
                        + duration_log_probs[parent_index * duration_count + state.duration_index];
            }
            if (!isfinite(hypothesis_scores[parent_index])
                || time_indexes[parent_index] >= output_length)
            {
                score = tdt_lowest_score();
            }
            candidate_scores[candidate] = isfinite(score) ? score : tdt_lowest_score();
            if (isfinite(score))
            {
                unsigned long long key =
                    state.hash
                    ^ (static_cast<unsigned long long>(state.length) * 0x9e3779b97f4a7c15ULL)
                    ^ (static_cast<unsigned long long>(state.time) * 0xbf58476d1ce4e5b9ULL)
                    ^ (static_cast<unsigned long long>(state.symbols) * 0x94d049bb133111ebULL);
                key ^= key >> 32;
                const int bucket = static_cast<unsigned int>(key) & (bucket_count - 1);
                candidate_links[candidate] = atomicExch(bucket_heads + bucket, candidate);
            }
        }
        __syncthreads();

        for (int bucket = thread; bucket < bucket_count; bucket += blockDim.x)
        {
            // Atomic insertion order varies between launches. Sort each short
            // bucket by candidate index so logaddexp and ties are reproducible.
            int sorted = -1;
            for (int candidate = bucket_heads[bucket]; candidate >= 0;)
            {
                const int next = candidate_links[candidate];
                if (sorted < 0 || candidate < sorted)
                {
                    candidate_links[candidate] = sorted;
                    sorted = candidate;
                }
                else
                {
                    int previous = sorted;
                    while (candidate_links[previous] >= 0 && candidate_links[previous] < candidate)
                    {
                        previous = candidate_links[previous];
                    }
                    candidate_links[candidate] = candidate_links[previous];
                    candidate_links[previous] = candidate;
                }
                candidate = next;
            }
            for (int first = sorted; first >= 0; first = candidate_links[first])
            {
                const TdtCandidate state = tdt_candidate(first, beam, duration_count,
                    positive_duration_count, hypothesis_base, blank_id, max_symbols_per_timestep,
                    output_length, durations, positive_duration_indexes, top_token_indexes,
                    hypothesis_hashes, hypothesis_lengths, time_indexes, symbols_at_timestep);
                float merged = candidate_scores[first];
                float best_score = merged;
                int best = first;
                int previous = first;
                for (int other = candidate_links[first]; other >= 0; other = candidate_links[other])
                {
                    const TdtCandidate other_state = tdt_candidate(other, beam, duration_count,
                        positive_duration_count, hypothesis_base, blank_id,
                        max_symbols_per_timestep, output_length, durations,
                        positive_duration_indexes, top_token_indexes, hypothesis_hashes,
                        hypothesis_lengths, time_indexes, symbols_at_timestep);
                    if (state.hash == other_state.hash && state.length == other_state.length
                        && state.time == other_state.time && state.symbols == other_state.symbols
                        && tdt_histories_equal(state, other_state, blank_id, hypothesis_nodes,
                            node_parents, node_tokens,
                            history_cache + static_cast<long long>(utterance) * history_cache_size,
                            history_cache_size))
                    {
                        const float score = candidate_scores[other];
                        merged = tdt_merge_log_scores(merged, score);
                        if (tdt_score_is_better(score, other, best_score, best))
                        {
                            best = other;
                            best_score = score;
                        }
                        candidate_scores[other] = tdt_lowest_score();
                        candidate_links[previous] = candidate_links[other];
                    }
                    else
                    {
                        previous = other;
                    }
                }
                candidate_scores[first] = tdt_lowest_score();
                candidate_scores[best] = merged;
            }
        }
        __syncthreads();

        for (int rank = 0; rank < beam; ++rank)
        {
            float best_score = tdt_lowest_score();
            int best_index = candidate_count;
            for (int candidate = thread; candidate < candidate_count; candidate += blockDim.x)
            {
                const float score = candidate_scores[candidate];
                if (tdt_score_is_better(score, candidate, best_score, best_index))
                {
                    best_score = score;
                    best_index = candidate;
                }
            }
            tdt_warp_best(best_score, best_index);
            if (lane == 0)
            {
                reduction_scores[warp] = best_score;
                reduction_indexes[warp] = best_index;
            }
            __syncthreads();

            if (warp == 0)
            {
                best_score = lane < num_warps ? reduction_scores[lane] : tdt_lowest_score();
                best_index = lane < num_warps ? reduction_indexes[lane] : candidate_count;
                tdt_warp_best(best_score, best_index);
                if (lane == 0)
                {
                    selected_scores[rank] = best_score;
                    selected_indexes[rank] = best_index;
                    candidate_scores[best_index] = tdt_removed_score();
                }
            }
            __syncthreads();
        }

        if (thread == 0)
        {
            const int node_base = utterance * beam * token_stride;
            int retained_count = 0;
            while (retained_count < beam && isfinite(selected_scores[retained_count]))
            {
                ++retained_count;
            }

            for (int output = 0; output < retained_count; ++output)
            {
                const int candidate = selected_indexes[output];
                const bool emitted = candidate < token_candidate_count;
                const TdtCandidate state = tdt_candidate(candidate, beam, duration_count,
                    positive_duration_count, hypothesis_base, blank_id, max_symbols_per_timestep,
                    output_length, durations, positive_duration_indexes, top_token_indexes,
                    hypothesis_hashes, hypothesis_lengths, time_indexes, symbols_at_timestep);
                const int parent_index = state.parent;

                int candidate_node = hypothesis_nodes[parent_index];
                if (emitted)
                {
                    // At most one node per retained beam item is allocated per
                    // search step. The host therefore reserves exactly
                    // beam * token_stride nodes for each utterance.
                    candidate_node = node_base + node_counts[utterance];
                    ++node_counts[utterance];
                    node_parents[candidate_node] = hypothesis_nodes[parent_index];
                    node_tokens[candidate_node] = state.token;
                    node_timestamps[candidate_node] =
                        roundf(time_indexes[parent_index] * encoder_frame_shift_sec * 1000.0F)
                        / 1000.0F;
                }

                const int output_index = hypothesis_base + output;
                next_scores[output_index] = selected_scores[output];
                next_nodes[output_index] = candidate_node;
                next_hashes[output_index] = state.hash;
                next_lengths[output_index] = state.length;
                next_time_indexes[output_index] = state.time;
                next_last_tokens[output_index] = emitted ? state.token : last_tokens[parent_index];
                next_symbols_at_timestep[output_index] = state.symbols;
                parent_indexes[output_index] = parent_index;
                use_output_state[output_index] = emitted;
            }

            int active_count = 0;
            for (int output = 0; output < retained_count; ++output)
            {
                const int output_index = hypothesis_base + output;
                const int length = next_lengths[output_index];
                if (next_time_indexes[output_index] >= output_length)
                {
                    if (!isfinite(completed_scores[utterance])
                        || tdt_length_normalized_score(next_scores[output_index], length + 1)
                               > tdt_length_normalized_score(
                                   completed_scores[utterance], completed_lengths[utterance] + 1))
                    {
                        completed_scores[utterance] = next_scores[output_index];
                        completed_nodes[utterance] = next_nodes[output_index];
                        completed_lengths[utterance] = length;
                    }
                    continue;
                }

                const int active_index = hypothesis_base + active_count;
                if (active_index != output_index)
                {
                    next_scores[active_index] = next_scores[output_index];
                    next_nodes[active_index] = next_nodes[output_index];
                    next_hashes[active_index] = next_hashes[output_index];
                    next_lengths[active_index] = length;
                    next_time_indexes[active_index] = next_time_indexes[output_index];
                    next_last_tokens[active_index] = next_last_tokens[output_index];
                    next_symbols_at_timestep[active_index] = next_symbols_at_timestep[output_index];
                    parent_indexes[active_index] = parent_indexes[output_index];
                    use_output_state[active_index] = use_output_state[output_index];
                }
                ++active_count;
            }

            for (int output = active_count; output < beam; ++output)
            {
                const int output_index = hypothesis_base + output;
                next_scores[output_index] = tdt_lowest_score();
                next_nodes[output_index] = -1;
                next_hashes[output_index] = 0;
                next_lengths[output_index] = 0;
                next_time_indexes[output_index] = output_length;
                next_last_tokens[output_index] = blank_id;
                next_symbols_at_timestep[output_index] = 0;
                parent_indexes[output_index] = -1;
                use_output_state[output_index] = 0;
            }
            active_flags[utterance] = active_count > 0;
        }
        __syncthreads();

        // Route each retained parent's recurrent state and prepare its next
        // encoder frame in the same kernel, avoiding two extra launches.
        for (int output = 0; output < beam; ++output)
        {
            gather_tdt_states(hypothesis_base + output, thread, input_state_1_raw,
                input_state_2_raw, output_state_1_raw, output_state_2_raw, parent_indexes,
                use_output_state, encoder_output_raw, next_scores, next_time_indexes,
                next_last_tokens, next_state_1_raw, next_state_2_raw, encoder_input_raw, targets,
                output_length, decoder_capacity, hidden_dim, state_layers, num_frames, encoder_dim,
                beam, state_dtype, encoder_output_dtype, encoder_input_dtype);
        }
    }

    __device__ __forceinline__ void gather_tdt_states(int hypothesis, int thread,
        const void* input_state_1_raw, const void* input_state_2_raw,
        const void* output_state_1_raw, const void* output_state_2_raw, const int* parent_indexes,
        const unsigned char* use_output_state, const void* encoder_output_raw,
        const float* hypothesis_scores, const int* time_indexes, const int* last_tokens,
        void* next_state_1_raw, void* next_state_2_raw, void* encoder_input_raw, int* targets,
        int output_length, int decoder_capacity, int hidden_dim, int state_layers, int num_frames,
        int encoder_dim, int beam, int state_dtype, int encoder_output_dtype,
        int encoder_input_dtype)
    {
        const int parent = parent_indexes[hypothesis];
        const bool output_state = use_output_state[hypothesis];
        if (parent < 0)
        {
            // Inactive slots remain score-masked, so their state and encoder rows
            // need not be rewritten. Only the predictor token must stay valid.
            if (thread == 0)
            {
                targets[hypothesis] = 0;
            }
            return;
        }
        // Route recurrent state from the predictor output after emission, or
        // preserve its input after a blank. A bitwise uint4 path moves 16 bytes
        // at once for every supported precision when rows are suitably aligned.
        const int packed_state_elements = state_dtype == TDT_FLOAT32 ? 4 : 8;
        const uintptr_t state_addresses = reinterpret_cast<uintptr_t>(input_state_1_raw)
                                          | reinterpret_cast<uintptr_t>(input_state_2_raw)
                                          | reinterpret_cast<uintptr_t>(output_state_1_raw)
                                          | reinterpret_cast<uintptr_t>(output_state_2_raw)
                                          | reinterpret_cast<uintptr_t>(next_state_1_raw)
                                          | reinterpret_cast<uintptr_t>(next_state_2_raw);
        if (hidden_dim % packed_state_elements == 0 && state_addresses % alignof(uint4) == 0)
        {
            const int hidden_vectors = hidden_dim / packed_state_elements;
            const uint4* input_state_1 = reinterpret_cast<const uint4*>(input_state_1_raw);
            const uint4* input_state_2 = reinterpret_cast<const uint4*>(input_state_2_raw);
            const uint4* output_state_1 = reinterpret_cast<const uint4*>(output_state_1_raw);
            const uint4* output_state_2 = reinterpret_cast<const uint4*>(output_state_2_raw);
            uint4* next_state_1 = reinterpret_cast<uint4*>(next_state_1_raw);
            uint4* next_state_2 = reinterpret_cast<uint4*>(next_state_2_raw);
            for (unsigned int state_vector = thread; state_vector < state_layers * hidden_vectors;
                state_vector += blockDim.x)
            {
                const int layer = state_vector / hidden_vectors;
                const int hidden_vector = state_vector - layer * hidden_vectors;
                const int destination =
                    (layer * decoder_capacity + hypothesis) * hidden_vectors + hidden_vector;
                const int source =
                    (layer * decoder_capacity + parent) * hidden_vectors + hidden_vector;
                next_state_1[destination] =
                    output_state ? output_state_1[source] : input_state_1[source];
                next_state_2[destination] =
                    output_state ? output_state_2[source] : input_state_2[source];
            }
        }
        else if (state_dtype != TDT_FLOAT32)
        {
            const unsigned short* input_state_1 =
                reinterpret_cast<const unsigned short*>(input_state_1_raw);
            const unsigned short* input_state_2 =
                reinterpret_cast<const unsigned short*>(input_state_2_raw);
            const unsigned short* output_state_1 =
                reinterpret_cast<const unsigned short*>(output_state_1_raw);
            const unsigned short* output_state_2 =
                reinterpret_cast<const unsigned short*>(output_state_2_raw);
            unsigned short* next_state_1 = reinterpret_cast<unsigned short*>(next_state_1_raw);
            unsigned short* next_state_2 = reinterpret_cast<unsigned short*>(next_state_2_raw);
            for (unsigned int state_index = thread; state_index < state_layers * hidden_dim;
                state_index += blockDim.x)
            {
                const int layer = state_index / hidden_dim;
                const int hidden = state_index - layer * hidden_dim;
                const int destination =
                    (layer * decoder_capacity + hypothesis) * hidden_dim + hidden;
                const int source = (layer * decoder_capacity + parent) * hidden_dim + hidden;
                next_state_1[destination] =
                    output_state ? output_state_1[source] : input_state_1[source];
                next_state_2[destination] =
                    output_state ? output_state_2[source] : input_state_2[source];
            }
        }
        else
        {
            const float* input_state_1 = reinterpret_cast<const float*>(input_state_1_raw);
            const float* input_state_2 = reinterpret_cast<const float*>(input_state_2_raw);
            const float* output_state_1 = reinterpret_cast<const float*>(output_state_1_raw);
            const float* output_state_2 = reinterpret_cast<const float*>(output_state_2_raw);
            float* next_state_1 = reinterpret_cast<float*>(next_state_1_raw);
            float* next_state_2 = reinterpret_cast<float*>(next_state_2_raw);
            for (unsigned int state_index = thread; state_index < state_layers * hidden_dim;
                state_index += blockDim.x)
            {
                const int layer = state_index / hidden_dim;
                const int hidden = state_index - layer * hidden_dim;
                const int destination =
                    (layer * decoder_capacity + hypothesis) * hidden_dim + hidden;
                const int source = (layer * decoder_capacity + parent) * hidden_dim + hidden;
                next_state_1[destination] =
                    output_state ? output_state_1[source] : input_state_1[source];
                next_state_2[destination] =
                    output_state ? output_state_2[source] : input_state_2[source];
            }
        }

        // Stage the retained hypothesis's next encoder frame and predictor token.
        const int utterance = hypothesis / beam;
        const bool active =
            isfinite(hypothesis_scores[hypothesis]) && time_indexes[hypothesis] < output_length;
        const int frame = min(time_indexes[hypothesis], num_frames - 1);
        const int output_base = (utterance * num_frames + frame) * encoder_dim;
        const int input_base = hypothesis * encoder_dim;
        stage_tdt_encoder_input(encoder_output_raw, encoder_input_raw, output_base, input_base,
            encoder_dim, encoder_output_dtype, encoder_input_dtype, active, thread);
        if (thread == 0)
        {
            targets[hypothesis] = active ? last_tokens[hypothesis] : 0;
        }
    }
    """,
    "tdt_beam_search",
    options=("--std=c++20",),
    backend="nvcc",
)

TDT_FINALIZE_KERNEL = cp.RawKernel(
    TDT_SCORE_HELPERS_SOURCE
    + r"""
    extern "C" __global__ void tdt_finalize(const float* hypothesis_scores,
        const int* hypothesis_nodes, const int* hypothesis_lengths, const float* completed_scores,
        const int* completed_nodes, const int* completed_lengths, const int* node_parents,
        const int* node_tokens, const float* node_timestamps, int* output_tokens,
        float* output_timestamps, int* output_token_lengths, int token_stride, int beam)
    {
        const int utterance = blockIdx.x;
        // Completed paths were ranked during search with the same token-count
        // normalization. If none completed, rank the surviving beam here.
        const bool use_completed = isfinite(completed_scores[utterance]);
        int selected_hypothesis = utterance * beam;
        int selected_node;
        int selected_length;
        if (!use_completed)
        {
            double selected_score =
                tdt_length_normalized_score(hypothesis_scores[selected_hypothesis],
                    hypothesis_lengths[selected_hypothesis] + 1);
            for (int hypothesis = 1; hypothesis < beam; ++hypothesis)
            {
                const int index = utterance * beam + hypothesis;
                const double score = tdt_length_normalized_score(
                    hypothesis_scores[index], hypothesis_lengths[index] + 1);
                if (score > selected_score)
                {
                    selected_hypothesis = index;
                    selected_score = score;
                }
            }
            selected_node = hypothesis_nodes[selected_hypothesis];
            selected_length = hypothesis_lengths[selected_hypothesis];
        }
        else
        {
            selected_node = completed_nodes[utterance];
            selected_length = completed_lengths[utterance];
        }
        output_token_lengths[utterance] = selected_length;
        // Histories are stored newest-first through parent links. Walk them
        // backward into chronological output order without a staging buffer.
        for (int position = selected_length - 1; position >= 0; --position)
        {
            const int output_index = utterance * token_stride + position;
            output_tokens[output_index] = node_tokens[selected_node];
            output_timestamps[output_index] = node_timestamps[selected_node];
            selected_node = node_parents[selected_node];
        }
    }
    """,
    "tdt_finalize",
    options=("--std=c++20",),
    backend="nvcc",
)
