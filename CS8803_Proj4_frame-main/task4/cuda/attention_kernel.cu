#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>



using namespace torch::indexing;

// FlashAttention kernel
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int B_r, int B_c>
__global__ void flashattention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int seq_len,
    int head_dim,
    bool causal)
{
    static_assert(B_c <= 32, "This kernel assumes one warp covers one K/V tile row.");

    // Shared memory for tiles
    __shared__ float q_tile[B_r][128];  // Max head_dim = 128
    __shared__ float k_tile[B_c][128];
    __shared__ float v_tile[B_c][128];
    __shared__ float m_tile[B_r];
    __shared__ float l_tile[B_r];
    __shared__ float p[B_r][B_c];

    // ########################################################

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;   // threadIdx.x / 32
    const int lane = tid & 31;      // threadIdx.x % 32
    const int bh_idx = blockIdx.y;
    const int q_start = static_cast<int>(blockIdx.x) * B_r;
    const int q_pos = q_start + warp_id;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    // Each lane accumulates up to 4 output elements (head_dim <= 128).
    float o_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Load and pre-scale one Q row per warp.
    if (warp_id < B_r) {
        for (int d = lane; d < head_dim; d += 32) {
            float q_val = 0.0f;
            if (q_pos < seq_len) {
                int q_offset = (bh_idx * seq_len + q_pos) * head_dim + d;
                q_val = q[q_offset];
            }
            q_tile[warp_id][d] = q_val * scale;
        }
        if (lane == 0) {
            m_tile[warp_id] = -INFINITY;
            l_tile[warp_id] = 0.0f;
        }
    }
    __syncthreads();

    const int num_kv_blocks = (seq_len + B_c - 1) / B_c;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        const int k_start = kv_block_idx * B_c;

        // Cooperatively load one K/V tile.
        for (int idx = tid; idx < B_c * head_dim; idx += blockDim.x) {
            int c = idx / head_dim;
            int d = idx % head_dim;
            int k_pos = k_start + c;
            if (k_pos < seq_len) {
                int kv_offset = (bh_idx * seq_len + k_pos) * head_dim + d;
                k_tile[c][d] = k[kv_offset];
                v_tile[c][d] = v[kv_offset];
            } else {
                k_tile[c][d] = 0.0f;
                v_tile[c][d] = 0.0f;
            }
        }
        __syncthreads();

        if (warp_id < B_r) {
            const bool row_valid = (q_pos < seq_len);

            // One lane computes one score in the current K/V tile.
            float score = -INFINITY;
            if (row_valid && lane < B_c) {
                int k_pos = k_start + lane;
                if (k_pos < seq_len && (!causal || k_pos <= q_pos)) {
                    float dot = 0.0f;
                    if (head_dim == 128) {
                        #pragma unroll
                        for (int d = 0; d < 128; d++) {
                            dot += q_tile[warp_id][d] * k_tile[lane][d];
                        }
                    } else {
                        for (int d = 0; d < head_dim; d++) {
                            dot += q_tile[warp_id][d] * k_tile[lane][d];
                        }
                    }
                    score = dot;
                }
            }

            float row_max = warp_reduce_max(score);
            row_max = __shfl_sync(0xffffffff, row_max, 0);

            float m_prev = m_tile[warp_id];
            float l_prev = l_tile[warp_id];
            float m_new = row_valid ? fmaxf(m_prev, row_max) : -INFINITY;

            float p_val = 0.0f;
            if (row_valid && isfinite(m_new)) {
                p_val = expf(score - m_new);
            }
            float row_sum = warp_reduce_sum(p_val);
            row_sum = __shfl_sync(0xffffffff, row_sum, 0);

            float alpha = 0.0f;
            if (row_valid) {
                alpha = isfinite(m_new) ? expf(m_prev - m_new) : 0.0f;
                if (lane == 0) {
                    l_tile[warp_id] = alpha * l_prev + row_sum;
                    m_tile[warp_id] = m_new;
                }
            }

            if (lane < B_c) {
                p[warp_id][lane] = p_val;
            }
            __syncwarp();

            // Compute/update output row in registers.
            if (row_valid) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int d = lane + i * 32;
                    if (d < head_dim) {
                        float pv = 0.0f;
                        #pragma unroll
                        for (int c = 0; c < B_c; c++) {
                            pv += p[warp_id][c] * v_tile[c][d];
                        }
                        o_reg[i] = alpha * o_reg[i] + pv;
                    }
                }
            }
        }

        // Ensure all warps are done reading this tile before overwriting it.
        __syncthreads();
    }

    if (warp_id < B_r) {
        if (q_pos < seq_len) {
            float inv_l = 1.0f / l_tile[warp_id];
            int out_offset = (bh_idx * seq_len + q_pos) * head_dim;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int d = lane + i * 32;
                if (d < head_dim) {
                    out[out_offset + d] = o_reg[i] * inv_l;
                }
            }
        }
    }

    // ########################################################
}

torch::Tensor custom_flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, int num_heads, bool causal) {
    auto options = q.options();
    
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int hidden_dim = q.size(2);
    int head_dim = hidden_dim / num_heads;
    
    // Reshape to (batch_size * num_heads, seq_len, head_dim)
    auto q_bh = q.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});
    auto k_bh = k.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});
    auto v_bh = v.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});
    
    auto out_bh = torch::zeros_like(q_bh, options);
    
    // ########################################################
    
    TORCH_CHECK(head_dim <= 128, "head_dim must be <= 128");

    // Warp-per-row launch configuration:
    // blockDim.x = B_r * 32 (one warp per Q row).
    if (seq_len >= 1024) {
        dim3 blocks((seq_len + 24 - 1) / 24, batch_size * num_heads);
        flashattention_kernel<24, 32><<<blocks, 24 * 32>>>(
            q_bh.data_ptr<float>(),
            k_bh.data_ptr<float>(),
            v_bh.data_ptr<float>(),
            out_bh.data_ptr<float>(),
            seq_len,
            head_dim,
            causal
        );
    } else if (seq_len >= 256) {
        dim3 blocks((seq_len + 16 - 1) / 16, batch_size * num_heads);
        flashattention_kernel<16, 32><<<blocks, 16 * 32>>>(
            q_bh.data_ptr<float>(),
            k_bh.data_ptr<float>(),
            v_bh.data_ptr<float>(),
            out_bh.data_ptr<float>(),
            seq_len,
            head_dim,
            causal
        );
    } else {
        dim3 blocks((seq_len + 8 - 1) / 8, batch_size * num_heads);
        flashattention_kernel<8, 32><<<blocks, 8 * 32>>>(
            q_bh.data_ptr<float>(),
            k_bh.data_ptr<float>(),
            v_bh.data_ptr<float>(),
            out_bh.data_ptr<float>(),
            seq_len,
            head_dim,
            causal
        );
    }
    
    // ########################################################

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Reshape back to (batch_size, seq_len, hidden_dim)
    auto out = out_bh.view({batch_size, num_heads, seq_len, head_dim})
                     .permute({0, 2, 1, 3})
                     .contiguous()
                     .view({batch_size, seq_len, hidden_dim});
    
    return out;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_flash_attention", &custom_flash_attention, "Custom FlashAttention in CUDA");
}
