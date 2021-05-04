// GPU-Globals
int d_warps;
unsigned int d_pitch;
real *c_syn0, *c_syn1neg;
constexpr real EXP_INDEX_SHIFT = EXP_TABLE_SIZE / (double)(MAX_EXP * 2.0);
dim3 cuda_blocks, cuda_threads;
unsigned int cuda_smem;

typedef unsigned int uint;
typedef int sint;
__global__ void neg_w2v_OUTER(real* syn0, real* syn1neg, const __restrict__ sint * const sen, const __restrict__ sint * const neg,  const __restrict__ sint * const length, const __restrict__ real * const alpha, const int d_layer_size, const int d_window, const int d_negative, const int d_MAX_SENTENCE_LENGTH, const int d_warps, const uint d_pitch, const int id);

void (*cuda_kernel)(real*, real*, const __restrict__ sint * const sen, const __restrict__ sint * const neg, const __restrict__ sint * const length, const __restrict__ real * const alpha, const int d_layer_size, const int d_window, const int d_negative, const int d_MAX_SENTENCE_LENGTH, const int d_warps, const uint d_pitch, const int id);

// Set appropriate launch conditions for the CUDA kernel and bind the proper kernel to the function variable
void kernel_setup(void) {
    int maxWords = negative + 1;
    cuda_blocks.x = kernel_batch_size;
    cuda_smem = 0;
    // Small kernel optimization
    if (1024 >= layer_size) {
        cuda_threads.x = layer_size;
        if (debug_mode > 0)
            fprintf(stderr, "\nKernel Selection: neg_w2v_OUTER -- GRID: <%d,%d,%d>\tBLOCK: <%d,%d,%d>\n", cuda_blocks.x, cuda_blocks.y, cuda_blocks.z, cuda_threads.x, cuda_threads.y, cuda_threads.z);
        cuda_kernel = &neg_w2v_OUTER;
    }
    else {
        cuda_threads.x = 1024;
        printf("SIZE %d > 1024 NOT SUPPORTED\n", layer_size);
        exit(1);
    }
    d_warps = (cuda_threads.x >> 5) + ((cuda_threads.x & 31) != 0);
}

#define PXL_LDG(g) __ldg(&(g))

#ifndef PXL_LDG
#warning "__ldg loads disabled for read-only values"
#define PXL_LDG(g) (g)
#else
#warning "__ldg loads enabled for most read-only values"
#endif

#if __CUDA_ARCH__ >= 320
#define PXL_LDG_IF(p,g) ((p) ? __ldg(&(g)) : (g))
#else
#define PXL_LDG_IF(p,g) (g)
#endif

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define PXL_GLOBAL_PTR   "l"
#else
#define PXL_GLOBAL_PTR   "r"
#endif

#if FALSE
#define DEVICE_STATIC_INTRINSIC_QUALIFIERS  static __device__ __forceinline__
DEVICE_STATIC_INTRINSIC_QUALIFIERS
void
__prefetch_global_l1(const void* const ptr)
{
  asm("prefetch.global.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

DEVICE_STATIC_INTRINSIC_QUALIFIERS
void
__prefetch_global_uniform(const void* const ptr)
{
  asm("prefetchu.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

DEVICE_STATIC_INTRINSIC_QUALIFIERS
void
__prefetch_global_l2(const void* const ptr)
{
  asm("prefetch.global.L2 [%0];" : : PXL_GLOBAL_PTR(ptr));
}
#endif

// All threads in the block collaborate to sum their input terms, but outputs differ as multiple words are computed simultaneously
__device__ __forceinline__
real blockReduceM2(real term, int laneID, int warpID, real *shared, int d_warps) {
    // Shuffle within warp
    term += __shfl_down_sync(0xFFFFFFFF, term, 16);
    term += __shfl_down_sync(0xFFFFFFFF, term, 8);
    term += __shfl_down_sync(0xFFFFFFFF, term, 4);
    term += __shfl_down_sync(0xFFFFFFFF, term, 2);
    term += __shfl_down_sync(0xFFFFFFFF, term, 1);
    if (!laneID) // Only 0th thread in warp writes to shared memory
      shared[warpID] = term;
    __syncthreads();
    term = 0.0f;
    #pragma unroll
    for (int i = 0; i < d_warps; ++i)
        term += shared[i];
    return term;
}

constexpr real dilute = 1.0f / (2.0f * MAX_EXP);
constexpr real expand = (2.0f * MAX_EXP * EXP_RESOLUTION);
__device__ __forceinline__ real sigmoid(real gradient, real sentence_alpha, const __restrict__ real* const exp)
{
    gradient = (gradient + MAX_EXP) * dilute;
    gradient = __saturatef(gradient);
    gradient = (gradient * expand);
    return sentence_alpha * exp[(int)(gradient)];
}

__device__ __forceinline__ void context_loop(int a, int b, real* syn0, const __restrict__ sint* const sen, __restrict__ real* block_shared, real& syn1neg_param, const int s_offset, const uint d_pitch, const int laneID, const int warpID, const int d_warps, const real sentence_alpha, const __restrict__ real* const exp)
{
    unsigned int syn0_index;
    real *syn0_vector, syn0_param, gradient;
    for (; a < b; a++) {
        // this chunk should already be in L1 because of syn1
        syn0_index = PXL_LDG(sen[s_offset + a]);

        syn0_vector = (syn0 + (syn0_index * d_pitch));
        syn0_param = syn0_vector[threadIdx.x];

        // Calculate gradient
        gradient = sigmoid(blockReduceM2(syn0_param * syn1neg_param, laneID, warpID, block_shared, d_warps), sentence_alpha, exp);

        // Calculate and apply updates
        atomicAdd(&syn0_vector[threadIdx.x], gradient * syn1neg_param);
        syn1neg_param += (gradient * syn0_param);
    }
}


__global__ void neg_w2v_OUTER(real* syn0, real* syn1neg, const __restrict__ sint * const sen, const __restrict__ sint * const neg, const __restrict__ sint * const length, const __restrict__ real * const alpha, const int d_layer_size, const int d_window, const int d_negative, const int d_MAX_SENTENCE_LENGTH, const int d_warps, const uint d_pitch, const int kernel_id) {
    // 8 warps is not supported on small kernel so this is more than enough
    __shared__ __restrict__ real block_shared[8];

    // Self-ID info
    register const int laneID = threadIdx.x & 0x1F,
                       warpID = threadIdx.x >> 5,
                       slen = PXL_LDG(length[blockIdx.x]),
                       s_offset = blockIdx.x * d_MAX_SENTENCE_LENGTH,
                       n_offset = s_offset * d_negative;

    //unsigned long long random = kernel_id;

    __restrict__ real* exp;
    real sentence_alpha;
	// Independence of negative samples
    for (int target = 0; target < d_negative+1; target++) {
        if (target) {
            sentence_alpha = PXL_LDG(alpha[blockIdx.x]);
            exp = nexpTable;
        }
        else {
            sentence_alpha = -PXL_LDG(alpha[blockIdx.x]);
            exp = expTable;
        }
    	// Consume target words from batch
        for (int w = 0; w < slen; w++) {
            int context_start = w - d_window,
                context_end = w + 1 + d_window;
            context_start = context_start < 0 ? 0 : context_start;
            context_end = context_end > slen ? slen : context_end;

            // Traverse window
            // BlockIdx.x == [0, Negatives+1] # Parallel negative ID, where 0 is context word rather than negative
            unsigned int syn1neg_index = target ? PXL_LDG(neg[n_offset + (w * d_negative) + (target-1)]) : PXL_LDG(sen[s_offset + w]);
            real *syn1neg_vector = (syn1neg + (syn1neg_index * d_pitch));
            // Cache param is not updated, used to determine aggregate update after window is trained
            real syn1neg_param = syn1neg_vector[threadIdx.x], cache_param = syn1neg_param;

            context_loop(context_start, w, syn0, sen, block_shared, syn1neg_param, s_offset, d_pitch, laneID, warpID, d_warps, sentence_alpha, exp);
            context_loop(1+w, context_end, syn0, sen, block_shared, syn1neg_param, s_offset, d_pitch, laneID, warpID, d_warps, sentence_alpha, exp);
            // Add accumulated difference in param to cached initial value
            atomicAdd(&syn1neg_vector[threadIdx.x], (syn1neg_param-cache_param));
        }
    }
}

__global__ void do_nothing(void) {
    return;
}

