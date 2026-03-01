#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>



using namespace torch::indexing;

// FlashAttention kernel
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
    // Shared memory for tiles
    __shared__ float q_tile[B_r][128];  // Max head_dim = 128
    __shared__ float k_tile[B_c][128];
    __shared__ float v_tile[B_c][128];
    __shared__ float o_tile[B_r][128];
    __shared__ float m_tile[B_r];
    __shared__ float l_tile[B_r];
    __shared__ float scores[B_r][B_c];
    __shared__ float p[B_r][B_c];

    // ########################################################

    const int tid = threadIdx.x;
    const int bh_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    const int q_start = q_block_idx * B_r;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    // Extra per-row shared state used by the online softmax update.
    __shared__ float alpha_tile[B_r];

    // Load Q tile for this Q block.
    for (int idx = tid; idx < B_r * head_dim; idx += blockDim.x) {
        int r = idx / head_dim;
        int d = idx % head_dim;
        int q_pos = q_start + r;
        if (q_pos < seq_len) {
            int q_offset = (bh_idx * seq_len + q_pos) * head_dim + d;
            q_tile[r][d] = q[q_offset];
        } else {
            q_tile[r][d] = 0.0f;
        }
    }

    // Initialize output accumulator and online-softmax statistics.
    for (int idx = tid; idx < B_r * head_dim; idx += blockDim.x) {
        int r = idx / head_dim;
        int d = idx % head_dim;
        o_tile[r][d] = 0.0f;
    }
    if (tid < B_r) {
        m_tile[tid] = -INFINITY;
        l_tile[tid] = 0.0f;
    }
    __syncthreads();

    const int num_kv_blocks = (seq_len + B_c - 1) / B_c;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        const int k_start = kv_block_idx * B_c;

        // Load K/V tiles.
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

        // Compute S = QK^T for this tile (with scale + optional causal mask).
        for (int idx = tid; idx < B_r * B_c; idx += blockDim.x) {
            int r = idx / B_c;
            int c = idx % B_c;
            int q_pos = q_start + r;
            int k_pos = k_start + c;

            float score = -INFINITY;
            if (q_pos < seq_len && k_pos < seq_len) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_tile[r][d] * k_tile[c][d];
                }
                score = dot * scale;
                if (causal && k_pos > q_pos) {
                    score = -INFINITY;
                }
            }
            scores[r][c] = score;
        }
        __syncthreads();

        // Update online-softmax stats and compute probabilities for this tile.
        if (tid < B_r) {
            int r = tid;
            int q_pos = q_start + r;

            if (q_pos < seq_len) {
                float row_max = -INFINITY;
                for (int c = 0; c < B_c; c++) {
                    row_max = fmaxf(row_max, scores[r][c]);
                }

                float m_prev = m_tile[r];
                float m_new = fmaxf(m_prev, row_max);
                float alpha = expf(m_prev - m_new);

                float row_sum = 0.0f;
                for (int c = 0; c < B_c; c++) {
                    float prob = expf(scores[r][c] - m_new);
                    p[r][c] = prob;
                    row_sum += prob;
                }

                alpha_tile[r] = alpha;
                l_tile[r] = alpha * l_tile[r] + row_sum;
                m_tile[r] = m_new;
            } else {
                alpha_tile[r] = 0.0f;
            }
        }
        __syncthreads();

        // Update output accumulator: O = alpha * O + P @ V
        for (int idx = tid; idx < B_r * head_dim; idx += blockDim.x) {
            int r = idx / head_dim;
            int d = idx % head_dim;
            int q_pos = q_start + r;
            if (q_pos < seq_len) {
                float pv = 0.0f;
                for (int c = 0; c < B_c; c++) {
                    pv += p[r][c] * v_tile[c][d];
                }
                o_tile[r][d] = alpha_tile[r] * o_tile[r][d] + pv;
            }
        }
        __syncthreads();
    }

    // Normalize by L and write output.
    for (int idx = tid; idx < B_r * head_dim; idx += blockDim.x) {
        int r = idx / head_dim;
        int d = idx % head_dim;
        int q_pos = q_start + r;
        if (q_pos < seq_len) {
            int out_offset = (bh_idx * seq_len + q_pos) * head_dim + d;
            out[out_offset] = o_tile[r][d] / l_tile[r];
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

    const int threads_per_block = 256;
    dim3 threads(threads_per_block);

    // Tune tile sizes: larger K/V tile helps long-sequence throughput while
    // staying under static shared-memory constraints on H100.
    if (seq_len >= 1024) {
        dim3 blocks((seq_len + 8 - 1) / 8, batch_size * num_heads);
        flashattention_kernel<8, 32><<<blocks, threads>>>(
            q_bh.data_ptr<float>(),
            k_bh.data_ptr<float>(),
            v_bh.data_ptr<float>(),
            out_bh.data_ptr<float>(),
            seq_len,
            head_dim,
            causal
        );
    } else {
        dim3 blocks((seq_len + 16 - 1) / 16, batch_size * num_heads);
        flashattention_kernel<16, 16><<<blocks, threads>>>(
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
