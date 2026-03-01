#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>

using namespace torch::indexing;

__global__ void softmax_kernel_batched(float *inp, float *outp, int NUM_ROW, int NUM_COL)
{
    extern __shared__ float buffer[];

    int b = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    if (row >= NUM_ROW) return;

    int row_offset = (b * NUM_ROW + row) * NUM_COL;

    float local_max = -FLT_MAX;
    for (int col = tid; col < NUM_COL; col += blockDim.x)
    {
        local_max = fmaxf(local_max, inp[row_offset + col]);
    }
    buffer[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            buffer[tid] = fmaxf(buffer[tid], buffer[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = buffer[0];

    float local_sum = 0.0f;
    for (int col = tid; col < NUM_COL; col += blockDim.x)
    {
        local_sum += expf(inp[row_offset + col] - row_max);
    }
    buffer[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            buffer[tid] += buffer[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = buffer[0];
    if (row_sum <= 0.0f)
    {
        row_sum = 1.0f;
    }

    for (int col = tid; col < NUM_COL; col += blockDim.x)
    {
        outp[row_offset + col] = expf(inp[row_offset + col] - row_max) / row_sum;
    }
}

template <int TILE_SIZE>
__global__ void GEMM_NT_kernel_batched(float *a_mat, float *b_mat, float *out_mat, int M, int N, int K)
{
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    int batch = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    int a_batch_offset = batch * M * K;
    int b_batch_offset = batch * N * K;
    int out_batch_offset = batch * M * N;

    float acc = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile)
    {
        int a_col = tile * TILE_SIZE + tx;
        int b_k = tile * TILE_SIZE + ty;

        if (row < M && a_col < K)
        {
            a_tile[ty][tx] = a_mat[a_batch_offset + row * K + a_col];
        }
        else
        {
            a_tile[ty][tx] = 0.0f;
        }

        if (col < N && b_k < K)
        {
            b_tile[ty][tx] = b_mat[b_batch_offset + col * K + b_k];
        }
        else
        {
            b_tile[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k_idx = 0; k_idx < TILE_SIZE; ++k_idx)
        {
            acc += a_tile[ty][k_idx] * b_tile[k_idx][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N)
    {
        out_mat[out_batch_offset + row * N + col] = acc;
    }
}

template <int TILE_SIZE>
__global__ void GEMM_NN_kernel_batched(float *a_mat, float *b_mat, float *out_mat, int M, int N, int K)
{
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    int batch = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    int a_batch_offset = batch * M * K;
    int b_batch_offset = batch * K * N;
    int out_batch_offset = batch * M * N;

    float acc = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile)
    {
        int a_col = tile * TILE_SIZE + tx;
        int b_row = tile * TILE_SIZE + ty;

        if (row < M && a_col < K)
        {
            a_tile[ty][tx] = a_mat[a_batch_offset + row * K + a_col];
        }
        else
        {
            a_tile[ty][tx] = 0.0f;
        }

        if (b_row < K && col < N)
        {
            b_tile[ty][tx] = b_mat[b_batch_offset + b_row * N + col];
        }
        else
        {
            b_tile[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k_idx = 0; k_idx < TILE_SIZE; ++k_idx)
        {
            acc += a_tile[ty][k_idx] * b_tile[k_idx][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N)
    {
        out_mat[out_batch_offset + row * N + col] = acc;
    }
}

__global__ void scale_and_causal_mask_batched(float *mat, int rows, int cols, float scale, bool causal)
{
    int b = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    if (row >= rows) return;

    int stride = rows * cols;
    float *row_ptr = mat + b * stride + row * cols;
    for (int j = tid; j < cols; j += blockDim.x)
    {
        float val = row_ptr[j] * scale;
        if (causal && j > row)
        {
            val = -FLT_MAX;
        }
        row_ptr[j] = val;
    }
}

torch::Tensor custom_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, int num_heads, bool causal)
{
    auto options = q.options();

    const int TILE_SIZE = 16;

    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int hidden_dim = q.size(2);
    int head_dim = hidden_dim / num_heads;
    int n_bh = batch_size * num_heads;

    // Reshape to (batch_size * num_heads, seq_len, head_dim) for parallel per-head processing
    auto q_bh = q.view({batch_size, seq_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous().view({n_bh, seq_len, head_dim});
    auto k_bh = k.view({batch_size, seq_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous().view({n_bh, seq_len, head_dim});
    auto v_bh = v.view({batch_size, seq_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous().view({n_bh, seq_len, head_dim});

    auto qk = torch::empty({n_bh, seq_len, seq_len}, options);
    auto s = torch::empty_like(qk);
    auto o_bh = torch::empty({n_bh, seq_len, head_dim}, options);

    dim3 gemm_threads(TILE_SIZE, TILE_SIZE);
    dim3 gemm_nt_blocks(
        (seq_len + TILE_SIZE - 1) / TILE_SIZE,
        (seq_len + TILE_SIZE - 1) / TILE_SIZE,
        n_bh);
    GEMM_NT_kernel_batched<TILE_SIZE><<<gemm_nt_blocks, gemm_threads>>>(
        q_bh.data_ptr<float>(),
        k_bh.data_ptr<float>(),
        qk.data_ptr<float>(),
        seq_len,
        seq_len,
        head_dim);

    dim3 scale_blocks(n_bh, seq_len);
    dim3 scale_threads(256);
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    scale_and_causal_mask_batched<<<scale_blocks, scale_threads>>>(
        qk.data_ptr<float>(),
        seq_len,
        seq_len,
        scale,
        causal);

    dim3 softmax_threads(256);
    dim3 softmax_blocks(n_bh, seq_len);
    softmax_kernel_batched<<<
        softmax_blocks,
        softmax_threads,
        softmax_threads.x * sizeof(float)>>>(
        qk.data_ptr<float>(),
        s.data_ptr<float>(),
        seq_len,
        seq_len);

    dim3 gemm_nn_blocks(
        (head_dim + TILE_SIZE - 1) / TILE_SIZE,
        (seq_len + TILE_SIZE - 1) / TILE_SIZE,
        n_bh);
    GEMM_NN_kernel_batched<TILE_SIZE><<<gemm_nn_blocks, gemm_threads>>>(
        s.data_ptr<float>(),
        v_bh.data_ptr<float>(),
        o_bh.data_ptr<float>(),
        seq_len,
        head_dim,
        seq_len);

    // Reshape back to (batch_size, seq_len, hidden_dim)
    auto o = o_bh.view({batch_size, num_heads, seq_len, head_dim}).permute({0, 2, 1, 3}).contiguous().view({batch_size, seq_len, hidden_dim});
    return o;
}

torch::Tensor test_batched_softmax(torch::Tensor inp, int dim) {
    int N_DIM = inp.dim();
    if (dim < 0) {
        dim += N_DIM;
    }
    if (dim != N_DIM - 1) {
        inp = inp.transpose(dim, N_DIM - 1);
    }
    // Ensure contiguous memory layout for linear indexing in the CUDA kernel
    inp = inp.contiguous();

    auto inp_sizes = inp.sizes();
    int N_BATCH = inp_sizes[0];
    int N_ROW = inp_sizes[1];
    int N_COL = inp_sizes[2];
    dim3 threads_per_block(256);
    dim3 blocks_per_grid(N_BATCH, N_ROW);
    auto outp = torch::empty(inp_sizes, inp.options());

    // Allocate shared memory for each thread in the block
    softmax_kernel_batched<<<blocks_per_grid, threads_per_block, threads_per_block.x * sizeof(float)>>>(
        inp.data_ptr<float>(), outp.data_ptr<float>(), N_ROW, N_COL);

    if (dim != N_DIM - 1) {
        inp = inp.transpose(dim, N_DIM - 1);
        outp = outp.transpose(dim, N_DIM - 1);
    }

    return outp;
}

torch::Tensor test_batched_GEMM_NN(torch::Tensor a, torch::Tensor b)
{
    a = a.contiguous();
    b = b.contiguous();
    auto a_size = a.sizes();
    auto b_size = b.sizes();
    int N_BATCH = a_size[0];
    int M = a_size[1], N = b_size[2], K = a_size[2];

    auto out = torch::empty({N_BATCH, M, N}, a.options());
    const int TILE_SIZE = 16;
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        N_BATCH);
    GEMM_NN_kernel_batched<TILE_SIZE><<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
    return out;
}

torch::Tensor test_batched_GEMM_NT(torch::Tensor a, torch::Tensor b)
{
    a = a.contiguous();
    b = b.contiguous();
    auto a_size = a.sizes();
    auto b_size = b.sizes();
    int N_BATCH = a_size[0];
    int M = a_size[1], N = b_size[1], K = a_size[2];

    auto out = torch::empty({N_BATCH, M, N}, a.options());
    const int TILE_SIZE = 16;
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        N_BATCH);
    GEMM_NT_kernel_batched<TILE_SIZE><<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
    return out;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_batched_softmax", &test_batched_softmax, "Testing interface for custom softmax in CUDA");
    m.def("test_batched_GEMM_NN", &test_batched_GEMM_NN, "Testing interface for custom matmul (A @ B) in CUDA");
    m.def("test_batched_GEMM_NT", &test_batched_GEMM_NT, "Testing interface for custom matmul (A @ B^T) in CUDA");
    m.def("custom_attention", &custom_attention, "Custom attention in CUDA");
}
