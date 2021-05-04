// GPU-Globals
int d_warps;
unsigned int d_pitch;
real *c_syn0, *c_syn1neg;
constexpr real EXP_INDEX_SHIFT = EXP_TABLE_SIZE / (double)(MAX_EXP * 2.0);
dim3 cuda_blocks, cuda_threads;
unsigned int cuda_smem;

typedef unsigned int uint;
typedef int sint;
__global__ void matrix_matrix_w2v_kernel_small(real* syn0, real* syn1neg, const __restrict__ sint * const sen, const __restrict__ sint * const neg,  const __restrict__ sint * const length, const __restrict__ real * const alpha, const int d_layer_size, const int d_window, const int d_negative, const int d_MAX_SENTENCE_LENGTH, const int d_warps, const uint d_pitch, const int id);
void (*cuda_kernel)(real*, real*, const __restrict__ sint * const sen, const __restrict__ sint * const neg, const __restrict__ sint * const length, const __restrict__ real * const alpha, const int d_layer_size, const int d_window, const int d_negative, const int d_MAX_SENTENCE_LENGTH, const int d_warps, const uint d_pitch, const int id);

// Set appropriate launch conditions for the CUDA kernel and bind the proper kernel to the function variable
void kernel_setup(void) {
    cuda_blocks.x = kernel_batch_size;
    int window_size = (2*window) + 1;
    cuda_smem = sizeof(float)*(8+(window_size)+(2*(layer_size * window_size)));
    // Small kernel optimization
    if (1024 >= layer_size) {
        cuda_threads.x = layer_size;
        if (debug_mode > 0)
            fprintf(stderr, "\nKernel Selection: matrix_matrix_w2v_kernel_small -- GRID: <%d,%d,%d>\tBLOCK: <%d,%d,%d>\n", cuda_blocks.x, cuda_blocks.y, cuda_blocks.z, cuda_threads.x, cuda_threads.y, cuda_threads.z);
        cuda_kernel = &matrix_matrix_w2v_kernel_small;
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
void __prefetch_global_l1(const void* const ptr) {
  asm("prefetch.global.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

DEVICE_STATIC_INTRINSIC_QUALIFIERS
void __prefetch_global_uniform(const void* const ptr) {
  asm("prefetchu.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

DEVICE_STATIC_INTRINSIC_QUALIFIERS
void __prefetch_global_l2(const void* const ptr) {
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
__device__ __forceinline__ real sigmoid(real gradient, real sentence_alpha, const __restrict__ real* const exp) {
    gradient = (gradient + MAX_EXP) * dilute;
    gradient = __saturatef(gradient);
    gradient = (gradient * expand);
    return sentence_alpha * exp[(int)(gradient)];
}

__device__ __forceinline__ int INDEX(int base, int wrap) {
  while (base >= wrap) base -= wrap;
  return base;
}

__global__ void matrix_matrix_w2v_kernel_small(real* syn0, real* syn1neg, const __restrict__ sint * const sen, const __restrict__ sint * const neg, const __restrict__ sint * const length, const __restrict__ real * const alpha, const int d_layer_size, const int d_window, const int d_negative, const int d_MAX_SENTENCE_LENGTH, const int d_warps, const uint d_pitch, const int kernel_id) {
    // Shared memory division between block reduction and caching memory
    extern __shared__ __restrict__ real block_shared[];
    // Self-ID info
    register const int laneID = threadIdx.x & 0x1F,
                       warpID = threadIdx.x >> 5,
                       slen = PXL_LDG(length[blockIdx.x]),
                       s_offset = blockIdx.x * d_MAX_SENTENCE_LENGTH,
                       n_offset = s_offset * d_negative,
                       window_size = (d_window<<1) + 1;

    float *block_reduce = block_shared,
          *vector_cache = (float*)&block_shared[8+window_size], // 2W+1 * layer_size floats for cache
          *OLDEST_CACHE = (float*)&block_shared[8+window_size+(window_size*d_layer_size)]; // Another 2W+1 * layer_size floats for static cache
    int *index_cache = (int*)&block_shared[8];
    // Ring buffer info
    int initialized = d_window + 1, // Used to eliminate words in warmup
        max_cache_position = -1; // For when the cache recognizes its end
    if (initialized > slen) {
        initialized = slen; // Bounds check
        max_cache_position = initialized - 1;
    }
    else if (window_size > slen) max_cache_position = slen - 1; // Alternative bounds check

	int cache_initialized = initialized, // Used for left window sizing
        touched = initialized, // Used to warm up deprecation
        position = initialized, // Controls insertion position
        center = 0; // Controls omit position
    if (slen < center) center = slen; // Ensure boundaries on small sentence initialization

    // Preload up to width+1 context words (size of initial context with target word)
    unsigned int syn0_index;
    real *syn0_vector, gradient;
    for (int w = 0; w < initialized; w++) {
        syn0_index = PXL_LDG(sen[s_offset + w]);
        index_cache[w] = syn0_index;
        syn0_vector = (syn0 + (syn0_index * d_pitch));
        // Coalesced read into shared memory (twice for caching purpose)
        OLDEST_CACHE[(w * d_layer_size)+threadIdx.x] = vector_cache[(w * d_layer_size)+threadIdx.x] = syn0_vector[threadIdx.x];
    }

    // Train the sentence
    __restrict__ real* exp;
    real sentence_alpha = -PXL_LDG(alpha[blockIdx.x]);
    for (int w = 0; w < slen; w++) {
      for (int target = 0; target < d_negative+1; target++) {
        unsigned int syn1neg_index;
        if (!target) { // Swap to target word
          sentence_alpha *= -1;
          exp = nexpTable;
          syn1neg_index = PXL_LDG(sen[s_offset + w]);
        }
        else { // Negative Sample
          if (target == 1) { // Switch to negatives (just once please)
            sentence_alpha *= -1;
            exp = expTable;
          }
          syn1neg_index = PXL_LDG(neg[n_offset + (w * d_negative) + (target-1)]);
        }
        // Independence of negative samples
        real *syn1neg_vector = (syn1neg + (syn1neg_index * d_pitch));
        real syn1neg_param = syn1neg_vector[threadIdx.x],
             cache_param = syn1neg_param;
        // Lifetime reuse of context words ring buffer
        if (touched == window_size && max_cache_position == -1) {
            for (int i = 0; i < center; i++) {
                real syn0_param = vector_cache[(i * d_layer_size) + threadIdx.x];
                gradient = sigmoid(blockReduceM2(syn0_param * syn1neg_param, laneID, warpID, block_reduce, d_warps), sentence_alpha, exp);
                vector_cache[(i * d_layer_size) + threadIdx.x] += gradient * syn1neg_param;
                syn1neg_param += (gradient * syn0_param);
            }
            for (int i = center+1; i < window_size; i++) {
                real syn0_param = vector_cache[(i * d_layer_size) + threadIdx.x];
                gradient = sigmoid(blockReduceM2(syn0_param * syn1neg_param, laneID, warpID, block_reduce, d_warps), sentence_alpha, exp);
                vector_cache[(i * d_layer_size) + threadIdx.x] += gradient * syn1neg_param;
                syn1neg_param += (gradient * syn0_param);
            }
        }
        else { // WARMUP AND COOLDOWN
            // LEFT HALF
            int start = center + d_window + 1,
                min_condition = cache_initialized - initialized;
            if (d_window < min_condition) min_condition = d_window;
            int stop = start + min_condition;
            for (int idx = start; idx < stop; idx++) {
              int circle_idx = INDEX(idx, touched);
              // Do training based on circle_idx
              real syn0_param = vector_cache[(circle_idx * d_layer_size) + threadIdx.x];
              gradient = sigmoid(blockReduceM2(syn0_param * syn1neg_param, laneID, warpID, block_reduce, d_warps), sentence_alpha, exp);
              vector_cache[(circle_idx * d_layer_size) + threadIdx.x] += gradient * syn1neg_param;
              syn1neg_param += (gradient * syn0_param);
            }
            // RIGHT HALF
            start = center + 1;
            stop = start + d_window;
            if (stop > slen) stop = slen;
            else if (max_cache_position > -1 && slen > window_size && max_cache_position - w >= d_window) stop = max_cache_position + (window_size * (max_cache_position < start));
            for (int idx = start; idx < stop; idx++) {
              int circle_idx = INDEX(idx, window_size);
              // Do training based on circle_idx
              real syn0_param = vector_cache[(circle_idx * d_layer_size) + threadIdx.x];
              gradient = sigmoid(blockReduceM2(syn0_param * syn1neg_param, laneID, warpID, block_reduce, d_warps), sentence_alpha, exp);
              vector_cache[(circle_idx * d_layer_size) + threadIdx.x] += gradient * syn1neg_param;
              syn1neg_param += (gradient * syn0_param);
            }
        }

        // Store accumulation of negative in global memory
        atomicAdd(&syn1neg_vector[threadIdx.x], (syn1neg_param-cache_param));
      }
      // After training the context, move to the next one with ring buffer
      if (max_cache_position == -1 || (slen <= window_size && touched < slen)) {
        // Apply lifetime update of context word
        if (initialized == 0) {
          syn0_vector = (syn0 + (index_cache[position] * d_pitch));
          atomicAdd(&syn0_vector[threadIdx.x],
                    (vector_cache[(position * d_layer_size)+threadIdx.x]-OLDEST_CACHE[(position * d_layer_size)+threadIdx.x]));
        }
		// Fetch next context word for lifetime reuse
        syn0_index = PXL_LDG(sen[s_offset + w + d_window + 1]);
        index_cache[position] = syn0_index;
        syn0_vector = (syn0 + (syn0_index * d_pitch));
        // Coalesced read into shared memory
        OLDEST_CACHE[(position * d_layer_size)+threadIdx.x] = vector_cache[(position * d_layer_size)+threadIdx.x] = syn0_vector[threadIdx.x];
        // Check for ending the insertion process
        if (w + d_window + 2 == slen) max_cache_position = position + 1; // Add +2 instead of +1 because you'll be fetching AFTER w++; marks end of cache
        else {
          // Increment position trackers
          position++;
          if (position == window_size) position = 0;
          if (touched < window_size) touched++;
        }
      }
      // Track center of the ring buffer
      center++;
      if (center == window_size) center = 0;
      if (initialized > 0) initialized--;
    }
	// Terminate lifetime reuse for cache on shutdown
    for (; position < center; position++) {
        syn0_vector = (syn0 + (index_cache[position] * d_pitch));
        atomicAdd(&syn0_vector[threadIdx.x],
                  (vector_cache[(position * d_layer_size)+threadIdx.x]-OLDEST_CACHE[(position * d_layer_size)+threadIdx.x]));
    }
}

__global__ void do_nothing(void) {
    return;
}

