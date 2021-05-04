#define NOOMP
#define SHUFFLE

#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sys/mman.h> // mmap()
#include <sys/stat.h> // Get size of file
#include <fcntl.h> // open()
#include <unistd.h> // close()
#include "safecuda.h" // cuda macros, cuda libs, stdio.h/stdlib.h

using namespace std;

#define MAX_STRING 800
#define EXP_TABLE_SIZE 1000
constexpr float MAX_EXP = 6;

typedef float real;
typedef unsigned int u_int;
typedef unsigned long long u_longlong;
typedef long long s_longlong;

// Timing defintions/macros
#ifndef NOOMP
    #include <omp.h>
    typedef double timer, time_val;
    #define timerTick(var) (var = omp_get_wtime())
    #define elapse(start, end) (end-start)
#else
    #include <pthread.h>
    #include <time.h>
    typedef struct timespec timer;
    typedef double time_val;
    #define timerTick(var) (clock_gettime(CLOCK_MONOTONIC_RAW, &var))
    #define elapse(start, end) ( ((time_val)end.tv_sec + (end.tv_nsec/(time_val)1e9)) - ((time_val)start.tv_sec + (start.tv_nsec/(time_val)1e9)) )
    pthread_barrier_t train_barrier;
    pthread_mutex_t word_count_lock;
#endif

struct vocab_word {
    u_int cn = 0;
    char *word;
    u_int used = 0, skipped = 0, negatived = 0;
};

timer start, endt;
short int binary = 0, debug_mode = 2;
bool recount = false, _export = true, cache_eos = false, pre_shuffle = false, epoch_shuffle = false;
int negative = 5, min_count = 5, num_threads = 12, streams_per_thread = 4, min_reduce = 1, epoch = 5, window = 5, batch_size = 11, kernel_batch_size = 200;
int vocab_max_size = 1000, vocab_size = 0, layer_size = 100, MAX_SENTENCE_LENGTH = 1000, low_word_count = 500;
u_longlong train_words = 0, file_size = 0, word_count_actual = 0;
real alpha = 0.025f, starting_alpha, sample = 1e-3f;
constexpr real EXP_RESOLUTION = (EXP_TABLE_SIZE - 1) / (double)(MAX_EXP * 2.0f);

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING], save_corpus_cache_file[MAX_STRING], read_corpus_cache_file[MAX_STRING];
char save_corpus_file[MAX_STRING];
constexpr int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
constexpr int UNIGRAM_TABLE_SIZE = 1e8;

struct vocab_word *vocab = NULL;
int *vocab_hash = NULL;
int *unigramTable;
#include "expTable.h" // (-200k wps to build for CPU build, so should be the GPU version)
real *Wih, *Woh;

// Additional items
real *subsampling_probability;
// G_stream = sentence values, G_sentences = pointers into values that can be shuffled
int **global_sentences = NULL, *global_stream = NULL;
u_longlong sentence_count;

// CUDA KERNEL SELECTION
#include "FULL-W2V-kernels.cu"
// Unchanged functions from Intel pWord2Vec.cpp (new/modified funcs defined in sub-included 'improvedPW2V_functions.cpp')
#include "legacy_driver_functions.cpp"

void * Train_SGNS(void *arg) {
    // Context fixing
    CUcontext context;
    CUdevice cuDevice;
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&context, cuDevice));
    cpu_set_t cpuset;

    #ifndef NOOMP
    #pragma omp parallel num_threads(num_threads)
    {
        int id = omp_get_thread_num();
    #else
        int id = *(int *)arg;
    #endif
        // Adjust scheduling affinity for perf
        CPU_ZERO(&cpuset); // clears the cpuset
        CPU_SET(id, &cpuset); // set CPU on cpuset
        // 0 is calling thread
        sched_setaffinity(0, sizeof(cpuset), &cpuset);

        // Determine training data bounds for the thread
        u_longlong epoch_start = 0, epoch_end, current_position = 0, stream_sentence = 0;
        {
            u_longlong per_thread = sentence_count / num_threads,
                       leftover = sentence_count % num_threads;
            for (u_longlong i = 0; i < (u_longlong)id; i++) epoch_start += per_thread + (i < leftover ? 1 : 0);
            epoch_end = epoch_start + per_thread + ((u_longlong)id < leftover ? 1 : 0);
            stream_sentence = epoch_start;
        }

        // Batching memory
        int local_epoch = epoch, sentences_batched = 0, sentence_length = 0, sentence_position = 0;
        u_longlong next_random = id, word_count = 0, last_word_count = 0;
        #ifdef CUDA_TALK
        char *cuda_msg = new char[100];
        #endif

        // CUDA streams
        u_longlong kernel_count = 0;
        int current_stream = 0;
        cudaStream_t* thread_stream = new cudaStream_t[streams_per_thread];
        for (int i = 0; i < streams_per_thread; i++) CHECK_CUDA_ERROR(cudaStreamCreate(&thread_stream[i]), cuda_msg, "TID %d: Claim CUDA stream %d", id, i);

        int **sen, **cuda_sen, **neg, **cuda_neg, **bsl, **cuda_bsl;
        real **b_alpha, **cuda_alpha;

        try {
            sen = new int* [streams_per_thread];
            cuda_sen = new int* [streams_per_thread];
            neg = new  int* [streams_per_thread];
            cuda_neg = new int* [streams_per_thread];
            bsl = new  int* [streams_per_thread];
            cuda_bsl = new int* [streams_per_thread];
            b_alpha = new real* [streams_per_thread];
            cuda_alpha = new real* [streams_per_thread];
        }
        catch (bad_alloc &ba) {
            cout << "Memory allocation failed: " << ba.what() << endl;
            exit(1);
        }

        for (int i = 0; i < streams_per_thread; i++) {
            CUDA_ASSERT(cudaMallocHost(&sen[i], kernel_batch_size * MAX_SENTENCE_LENGTH * sizeof(int)), cuda_msg, "Thread %d Pinned allocation <sen %d>", id, i);
            CUDA_ASSERT(cudaMalloc(&cuda_sen[i], kernel_batch_size * MAX_SENTENCE_LENGTH * sizeof(int)), cuda_msg, "Thread %d CUDA allocation <sen %d>", id, i);
            CUDA_ASSERT(cudaMallocHost(&neg[i], kernel_batch_size * MAX_SENTENCE_LENGTH * negative * sizeof(int)), cuda_msg, "Thread %d Pinned allocation <neg %d>", id, i);
            CUDA_ASSERT(cudaMalloc(&cuda_neg[i], kernel_batch_size * MAX_SENTENCE_LENGTH * negative * sizeof(int)), cuda_msg, "Thread %d CUDA allocation <neg %d>", id, i);

            CUDA_ASSERT(cudaMallocHost(&bsl[i], kernel_batch_size * sizeof(int)), cuda_msg, "Thread %d Pinned allocation <sentence_length %d>", id, i);
            CUDA_ASSERT(cudaMalloc(&cuda_bsl[i], kernel_batch_size * sizeof(int)), cuda_msg, "Thread %d CUDA allocation <sentence_length %d>", id, i);

            CUDA_ASSERT(cudaMallocHost(&b_alpha[i], kernel_batch_size * sizeof(real)), cuda_msg, "Thread %d Pinned allocation <alpha %d>", id, i);
            CUDA_ASSERT(cudaMalloc(&cuda_alpha[i], kernel_batch_size * sizeof(real)), cuda_msg, "Thread %d CUDA allocation <alpha %d>", id, i);
        }

        #ifndef NOOMP
        #pragma omp barrier
        #else
        pthread_barrier_wait(&train_barrier);
        #endif

        if (id == 0) timerTick(start);
        timer epoch_start_t;
        timerTick(epoch_start_t);

        while (local_epoch > 0) {
            // Debug and learning rate update
            if (word_count - last_word_count > 10000) {
                u_longlong diff = word_count - last_word_count;
                #ifndef NOOMP
                #pragma omp atomic
                word_count_actual += diff;
                #else
                pthread_mutex_lock(&word_count_lock);
                word_count_actual += diff;
                pthread_mutex_unlock(&word_count_lock);
                #endif

                last_word_count = word_count;
                if (debug_mode > 0) {
                    timer now;
                    timerTick(now);
                    printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk -- kernel %llu", 13, alpha,
                            word_count_actual / (real) (epoch * train_words + 1) * 100,
                            (word_count_actual / 1000.0) / elapse(start, now), kernel_count);
                    fflush(stdout);
                }
                alpha = starting_alpha * (1 - word_count_actual / (real) (epoch * train_words + 1));
                if (alpha < starting_alpha * 0.0001f) alpha = starting_alpha * 0.0001f;
            }

            // New batch needed
            if (sentence_length == 0) {
                while (sentence_length < MAX_SENTENCE_LENGTH) {
                    // Change sentences due to MSL (subsampling messes with clean sentence bounds)
                    if (current_position >= (u_longlong)MAX_SENTENCE_LENGTH) {
                        stream_sentence++;
                        current_position = 0;
                        if (stream_sentence >= epoch_end) break;
                    }

                    // Select next word
                    s_longlong w = global_sentences[stream_sentence][current_position];
                    current_position++;
                    word_count++;

                    // Epoch end and EOS token arbitrarily end batches
                    if (w == 0 || stream_sentence >= epoch_end) {
                        // Need to select next sentence
                        stream_sentence++;
                        current_position = 0;
                        break;
                    }

                    // Subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        next_random = next_random * (u_longlong) 25214903917 + 11;
                        if (subsampling_probability[w] < (next_random & 0xFFFF) / 65536.f) continue;
                    }

                    // Add word to the trainable sentence
                    sen[current_stream][(sentences_batched * MAX_SENTENCE_LENGTH) + sentence_length] = w;
                    // Batch negatives (each target word has associated negatives)
                    for (int k = 0; k < negative; k++) {
                        next_random = next_random * (u_longlong) 25214903917 + 11;
                        neg[current_stream][(sentences_batched * MAX_SENTENCE_LENGTH * negative) + (sentence_length * negative) + k] = unigramTable[(next_random >> 16) % UNIGRAM_TABLE_SIZE];
                    }
                    sentence_length++;
                }
                sentence_position = 0;

                // Batched to end of epoch
                if (stream_sentence >= epoch_end) {
                    u_longlong diff = word_count - last_word_count;
                    #ifndef NOOMP
                    #pragma omp atomic
                    word_count_actual += diff;
                    #else
                    pthread_mutex_lock(&word_count_lock);
                    word_count_actual += diff;
                    pthread_mutex_unlock(&word_count_lock);
                    #endif

                    local_epoch--;
                    word_count = 0;
                    last_word_count = 0;
                    sentence_length = 0;
                    stream_sentence = epoch_start;
                    #ifdef SHUFFLE
                    if (epoch_shuffle) {
                        // Wait for all threads to finish the epoch
                        #ifndef NOOMP
                        #pragma omp barrier
                        #else
                        pthread_barrier_wait(&train_barrier);
                        #endif
                        // Thread 0 will shuffle
                        if (id == 0 && local_epoch > 0) shuffleSentences();
                        // Ready to begin new epoch
                        #ifndef NOOMP
                        #pragma omp barrier
                        #else
                        pthread_barrier_wait(&train_barrier);
                        #endif
                    }
                    #endif
                    timer now;
                    timerTick(now);
                    /*
                    if (debug_mode > 0)	{
						printf("\nThread %d End of Epoch #%d (%f s) Words/sec: %.2fk", id, epoch-local_epoch, elapse(epoch_start_t, now),
                            (word_count_actual / 1000.0) / elapse(start, now));
                    	fflush(stdout);
					}
                    */
                    epoch_start_t = now;
                }
            }

			// Store Metadata
            bsl[current_stream][sentences_batched] = sentence_length;
            b_alpha[current_stream][sentences_batched] = alpha;
            sentences_batched++;
            if (sentences_batched == kernel_batch_size) {
                // Launch the CUDA kernel
                CHECK_CUDA_ERROR(cudaStreamSynchronize(thread_stream[current_stream]), cuda_msg, "Thread %d prep memcopy syncs stream %d", id, current_stream);
                CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_sen[current_stream], sen[current_stream], sentences_batched * MAX_SENTENCE_LENGTH * sizeof(unsigned int), cudaMemcpyHostToDevice, thread_stream[current_stream]), cuda_msg, "Thread %d Xfer sentences to stream %d", id, current_stream);
                CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_neg[current_stream], neg[current_stream], sentences_batched * MAX_SENTENCE_LENGTH * negative * sizeof(unsigned int), cudaMemcpyHostToDevice, thread_stream[current_stream]), cuda_msg, "Thread %d Xfer negatives to stream %d", id, current_stream);
                CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_bsl[current_stream], bsl[current_stream], sentences_batched * sizeof(int), cudaMemcpyHostToDevice, thread_stream[current_stream]), cuda_msg, "Thread %d Xfer sentence lengths to stream %d", id, current_stream);
                CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_alpha[current_stream], b_alpha[current_stream], sentences_batched * sizeof(real), cudaMemcpyHostToDevice, thread_stream[current_stream]), cuda_msg, "Thread %d Xfer alphas to stream %d", id, current_stream);
                // Kernel launch
                CHECK_CUDA_ERROR(cudaStreamSynchronize(thread_stream[current_stream]), cuda_msg, "Thread %d prep kernel launch syncs stream %d", id, current_stream);
                cuda_kernel<<<cuda_blocks, cuda_threads, cuda_smem, thread_stream[current_stream]>>>(c_syn0, c_syn1neg, cuda_sen[current_stream], cuda_neg[current_stream], cuda_bsl[current_stream], cuda_alpha[current_stream], layer_size, window, negative, MAX_SENTENCE_LENGTH, d_warps, d_pitch, kernel_count);
                kernel_count++;
                sentences_batched = 0;
                // Select next stream
                current_stream++;
                if (current_stream == streams_per_thread) current_stream = 0;
            }
            // Mark batch as completely consumed
            sentence_length = 0;
        }

        // Final kernel to wrap up remaining work
        if (sentence_position >= sentence_length || sentences_batched > 0) {
            bsl[current_stream][sentences_batched] = sentence_length;
            b_alpha[current_stream][sentences_batched] = alpha;
            sentences_batched += 1;
            // Launch the CUDA kernel
            CHECK_CUDA_ERROR(cudaStreamSynchronize(thread_stream[current_stream]), cuda_msg, "Thread %d prep memcopy syncs stream %d", id, current_stream);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_sen[current_stream], sen[current_stream], sentences_batched * MAX_SENTENCE_LENGTH * sizeof(unsigned int), cudaMemcpyHostToDevice, thread_stream[current_stream]), cuda_msg, "Thread %d Xfer sentences to stream %d", id, current_stream);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_neg[current_stream], neg[current_stream], sentences_batched * MAX_SENTENCE_LENGTH * negative * sizeof(unsigned int), cudaMemcpyHostToDevice, thread_stream[current_stream]), cuda_msg, "Thread %d Xfer negatives to stream %d", id, current_stream);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_bsl[current_stream], bsl[current_stream], sentences_batched * sizeof(int), cudaMemcpyHostToDevice, thread_stream[current_stream]), cuda_msg, "Thread %d Xfer sentence lengths to stream %d", id, current_stream);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_alpha[current_stream], b_alpha[current_stream], sentences_batched * sizeof(real), cudaMemcpyHostToDevice, thread_stream[current_stream]), cuda_msg, "Thread %d Xfer alphas to stream %d", id, current_stream);
            // Kernel launch
            CHECK_CUDA_ERROR(cudaStreamSynchronize(thread_stream[current_stream]), cuda_msg, "Thread %d prep kernel launch syncs stream %d", id, current_stream);
            cuda_kernel<<<cuda_blocks, cuda_threads, cuda_smem, thread_stream[current_stream]>>>(c_syn0, c_syn1neg, cuda_sen[current_stream], cuda_neg[current_stream], cuda_bsl[current_stream], cuda_alpha[current_stream], layer_size, window, negative, MAX_SENTENCE_LENGTH, d_warps, d_pitch, kernel_count);
            kernel_count++;
        }
        if (debug_mode > 0 && id == 0) printf("\nFinal kernel count (per-thread): %llu\n", kernel_count);

        pthread_barrier_wait(&train_barrier);
        cudaDeviceSynchronize();
        if (id == 0) timerTick(endt);

        for (int i = 0; i < streams_per_thread; i++) {
            //CHECK_CUDA_ERROR(cudaStreamSynchronize(thread_stream[i]));
            CHECK_CUDA_ERROR(cudaFreeHost(sen[i]));
            CHECK_CUDA_ERROR(cudaFreeHost(neg[i]));
            CHECK_CUDA_ERROR(cudaFreeHost(bsl[i]));
            CHECK_CUDA_ERROR(cudaFreeHost(b_alpha[i]));
            CHECK_CUDA_ERROR(cudaFree(cuda_sen[i]));
            CHECK_CUDA_ERROR(cudaFree(cuda_neg[i]));
            CHECK_CUDA_ERROR(cudaFree(cuda_bsl[i]));
            CHECK_CUDA_ERROR(cudaFree(cuda_alpha[i]));
            CHECK_CUDA_ERROR(cudaStreamDestroy(thread_stream[i]), cuda_msg, "TID %d: Relinquish CUDA stream %d", id, i);
        }
        delete[] thread_stream;
        delete[] sen;
        delete[] neg;
        delete[] bsl;
        delete[] b_alpha;
        delete[] cuda_sen;
        delete[] cuda_neg;
        delete[] cuda_bsl;
        delete[] cuda_alpha;
        #ifdef CUDA_TALK
        delete[] cuda_msg;
        #endif
    #ifndef NOOMP
    }
    #endif
    return NULL;
}

void SetupNetwork() {
    // Vocabulary determination
    vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *) malloc(vocab_hash_size * sizeof(int));
    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    if (train_file[0] != 0) {
        const char *train_mapping;
        int fd = open(train_file, O_RDONLY);
        struct stat fd_stats;
        if (fstat(fd, &fd_stats) < 0) {
            printf("ERROR: Training data not found!\n");
            exit(1);
        }
        file_size = fd_stats.st_size;
        if (debug_mode > 0) printf("Train file size = %lluK\n", file_size / 1000);
        train_mapping = (char *) mmap(0, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);

        if (read_vocab_file[0] != 0)
            ReadVocab();
        else
            LearnVocabFromTrainFile(train_mapping);
        if (recount)
            recountVocab(train_mapping);
        if (save_vocab_file[0] != 0)
            SaveVocab();
        // Cache training data in shuffle-able array
        if (read_corpus_cache_file[0] != 0)
            readCorpusCache();
        else {
            try {
                global_stream = new int[train_words + 1];
            }
            catch (bad_alloc &ba) {
                cout << "Memory allocation failed: " << ba.what() << endl;
                exit(1);
            }
            global_sentences = (int **) malloc((vocab[0].cn + 1) * sizeof(int *));
            sentence_count = createGlobalStream(vocab[0].cn + 1, train_mapping);
        }
        munmap((void *)train_mapping, file_size);
    }
    else {
        if (read_vocab_file[0] == 0) {
            printf("ERROR: No vocabulary or training file to generate vocabulary from! Please specify -train or -read-vocab\n");
            exit(1);
        }
        ReadVocab();
        // Due to call to SortVocab(), new vocabulary ~= file vocab and may be saved
        if (save_vocab_file[0] != 0) SaveVocab();

        // Cache training data in shuffle-able array
        if (read_corpus_cache_file[0] == 0) {
            printf("ERROR: No corpus cache or training file to learn from! Please specify -train or -read-corpus-cache\n");
            exit(1);
        }
        readCorpusCache();
    }

    if (save_corpus_cache_file[0] != 0) saveCorpusCache();
    if (save_corpus_file[0] != 0) saveCorpus();

    if (debug_mode > 1) printf("%llu sentences spread across %d CPU threads == %llu sentences/thread\n", sentence_count, num_threads, sentence_count / num_threads);
    if (output_file[0] == 0) exit(0);
    InitNet();
    #ifdef SHUFFLE
    if (pre_shuffle) shuffleSentences();
    #endif
}

int main(int argc, char **argv) {
    setCommandLineValues(argc, argv);
    SetupNetwork();

    do_nothing<<<1, 1>>>(); // Some metrics best collected after device has run at least one kernel
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    kernel_setup();

    #ifndef NOOMP
    Train_SGNS(NULL);
    #else
    int *tids = new int[num_threads];
    pthread_barrier_init(&train_barrier, NULL, num_threads);
    pthread_t *threads = new pthread_t[num_threads];
    for (int i = 0; i < num_threads; i++) {
        tids[i] = i;
        pthread_create(&threads[i], NULL, Train_SGNS, &tids[i]);
    }
    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);
    #endif

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("\nOverall Words Per Second: %.4f\n", word_count_actual / elapse(start, endt));
    printf("Overall Words: %llu\nOverall time: %f\n", word_count_actual, elapse(start, endt));

    free(vocab_hash);
    //free(global_sentences);
    delete[] global_stream;
    if (sample > 0)
        delete[] subsampling_probability;
    if (_export)
        saveModel();
    #ifdef NOOMP
    pthread_barrier_destroy(&train_barrier);
    delete[] tids;
    delete[] threads;
    #endif
    // Free memory after exporting
    free(vocab);
    delete[] Wih;
    delete[] Woh;
    CHECK_CUDA_ERROR(cudaFree(c_syn0), "Free GPU syn0");
    CHECK_CUDA_ERROR(cudaFree(c_syn1neg), "Free GPU syn1neg");
    return 0;
}
