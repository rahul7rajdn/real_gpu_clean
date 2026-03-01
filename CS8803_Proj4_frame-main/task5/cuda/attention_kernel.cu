#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>


using namespace torch::indexing;

// FlashAttention decode kernel (q sequence length is always 1)
template <int B>
__global__ void flash_attention_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int seq_len,
    int head_dim,
    bool causal)
{
    // Shared memory for tiles
    __shared__ float q_tile[128];  // Max head_dim = 128
    __shared__ float k_tile[B][128];
    __shared__ float v_tile[B][128];
    __shared__ float o_tile[128];
    __shared__ float m_tile;
    __shared__ float l_tile;
    __shared__ float scores[B];
    __shared__ float p[B];

    // ########################################################

    const int tid = static_cast<int>(threadIdx.x);
    const int bh_idx = static_cast<int>(blockIdx.x);
    const float scale = rsqrtf(static_cast<float>(head_dim));

    // Load one query vector and initialize accumulators.
    if (tid < head_dim) {
        int q_offset = (bh_idx * 1 + 0) * head_dim + tid;
        q_tile[tid] = q[q_offset] * scale;
        o_tile[tid] = 0.0f;
    }
    if (tid == 0) {
        m_tile = -INFINITY;
        l_tile = 0.0f;
    }
    __syncthreads();

    const int num_kv_blocks = (seq_len + B - 1) / B;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        const int k_start = kv_block_idx * B;

        // Load K/V tiles.
        for (int idx = tid; idx < B * head_dim; idx += blockDim.x) {
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

        // Compute one score per K row.
        if (tid < B) {
            int k_pos = k_start + tid;
            float score = -INFINITY;
            if (k_pos < seq_len) {
                // q is the newest token and k contains only valid context [0..seq_len-1].
                if (!causal || k_pos <= (seq_len - 1)) {
                    float dot = 0.0f;
                    if (head_dim == 128) {
                        #pragma unroll
                        for (int d = 0; d < 128; d++) {
                            dot += q_tile[d] * k_tile[tid][d];
                        }
                    } else {
                        for (int d = 0; d < head_dim; d++) {
                            dot += q_tile[d] * k_tile[tid][d];
                        }
                    }
                    score = dot;
                }
            }
            scores[tid] = score;
        }
        __syncthreads();

        // Online softmax update (single row).
        if (tid == 0) {
            float row_max = -INFINITY;
            #pragma unroll
            for (int c = 0; c < B; c++) {
                row_max = fmaxf(row_max, scores[c]);
            }

            float m_new = fmaxf(m_tile, row_max);
            float alpha = expf(m_tile - m_new);

            float row_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < B; c++) {
                float prob = expf(scores[c] - m_new);
                p[c] = prob;
                row_sum += prob;
            }

            l_tile = alpha * l_tile + row_sum;
            m_tile = m_new;

            // Reuse scores[0] to broadcast alpha.
            scores[0] = alpha;
        }
        __syncthreads();

        // O = alpha * O + P @ V
        float alpha = scores[0];
        if (tid < head_dim) {
            float pv = 0.0f;
            #pragma unroll
            for (int c = 0; c < B; c++) {
                pv += p[c] * v_tile[c][tid];
            }
            o_tile[tid] = alpha * o_tile[tid] + pv;
        }
        __syncthreads();
    }

    // Normalize and write output.
    if (tid < head_dim) {
        int out_offset = (bh_idx * 1 + 0) * head_dim + tid;
        out[out_offset] = o_tile[tid] / l_tile;
    }

    // ########################################################
}

torch::Tensor custom_flash_attention_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, int num_heads, bool causal) {
    auto options = q.options();
    
    int batch_size = q.size(0);
    int seq_len = k.size(1);
    int hidden_dim = q.size(2);
    int head_dim = hidden_dim / num_heads;
    
    // Reshape to (batch_size * num_heads, 1, head_dim)
    auto q_bh = q.view({batch_size, 1, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, 1, head_dim});
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

    const int total_bh = batch_size * num_heads;
    dim3 blocks(total_bh);
    const int threads_per_block = 128;

    if (seq_len >= 128) {
        flash_attention_decode_kernel<32><<<blocks, threads_per_block>>>(
            q_bh.data_ptr<float>(),
            k_bh.data_ptr<float>(),
            v_bh.data_ptr<float>(),
            out_bh.data_ptr<float>(),
            seq_len,
            head_dim,
            causal
        );
    } else {
        flash_attention_decode_kernel<16><<<blocks, threads_per_block>>>(
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
    
    // Reshape back to (batch_size, 1, hidden_dim)
    auto out = out_bh.view({batch_size, num_heads, 1, head_dim})
                     .permute({0, 2, 1, 3})
                     .contiguous()
                     .view({batch_size, 1, hidden_dim});
    
    return out;
}

__global__ void update_cache_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int batch_size,
    int current_pos,
    int max_seq_len,
    int hidden_dim
) {
    // ########################################################

    int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    int total = batch_size * hidden_dim;
    if (idx >= total || current_pos < 0 || current_pos >= max_seq_len) {
        return;
    }

    int b = idx / hidden_dim;
    int d = idx % hidden_dim;

    int src_offset = b * hidden_dim + d;  // k/v are shaped [batch_size, 1, hidden_dim]
    int cache_offset = (b * max_seq_len + current_pos) * hidden_dim + d;

    k_cache[cache_offset] = k[src_offset];
    v_cache[cache_offset] = v[src_offset];

    // ########################################################
}

void update_kv_cache(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k, torch::Tensor v, int current_pos) {
    auto options = k_cache.options();

    int batch_size = k_cache.size(0);
    int max_seq_len = k_cache.size(1);
    int hidden_dim = k_cache.size(2);
    
    int total_threads = batch_size * hidden_dim;

    int threads_per_block = 256;
    dim3 threads(threads_per_block);
    dim3 blocks((total_threads + threads_per_block - 1) / threads_per_block);

    update_cache_kernel<<<blocks, threads>>>(
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        batch_size,
        current_pos,
        max_seq_len,
        hidden_dim
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_flash_attention_decode", &custom_flash_attention_decode, "Custom FlashAttention in CUDA");
    m.def("update_kv_cache", &update_kv_cache, "Update KV cache in CUDA");
}
