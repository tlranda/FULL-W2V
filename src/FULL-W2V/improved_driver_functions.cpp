void InitNet() {
    if (debug_mode > 0)
        fprintf(stderr, "Initializing network\n");

    // GPU-side model creation/initialization
    u_longlong model_size = vocab_size * layer_size;
    #ifdef CUDA_TALK
    char cuda_msg[1000];
    #endif
    size_t model_pitch;
    CUDA_ASSERT(cudaMallocPitch(&c_syn0, &model_pitch, layer_size * sizeof(real), vocab_size), cuda_msg, "GPU mem alloc for syn0");
    CUDA_ASSERT(cudaMallocPitch(&c_syn1neg, &model_pitch, layer_size * sizeof(real), vocab_size), cuda_msg, "GPU mem alloc for syn1neg");

    // Could become cudaMallocHost(), but deferred to threads (also not pinging model back and forth, so bandwidth noncritical)
    try {
        Wih = new real[model_size];
        Woh = new real[model_size];
    }
    catch (bad_alloc &ba) {
        cout << "Memory allocation failed: " << ba.what() << endl;
        exit(1);
    }
    #pragma omp parallel for num_threads(num_threads) schedule(static, 1)
    for (int i = 0; i < vocab_size; i++) memset(Woh + i * layer_size, 0.f, layer_size * sizeof(real));
    CHECK_CUDA_ERROR(cudaMemcpy2DAsync(c_syn1neg, model_pitch, Woh, layer_size * sizeof(real), layer_size * sizeof(real), vocab_size, cudaMemcpyHostToDevice), "Woh->syn1neg");
    // Wih initialization
    u_longlong next_random = 1;
    for (u_longlong i = 0; i < model_size; i++) {
        next_random = next_random * (u_longlong) 25214903917 + 11;
        Wih[i] = (((next_random & 0xFFFF) / 65536.f) - 0.5f) / layer_size;
    }
    CHECK_CUDA_ERROR(cudaMemcpy2DAsync(c_syn0, model_pitch, Wih, layer_size * sizeof(real), layer_size * sizeof(real), vocab_size, cudaMemcpyHostToDevice), "Wih->syn0");
    d_pitch = model_pitch/sizeof(real);

    starting_alpha = alpha;
    // Table used for selecting negative samples
    try {
        unigramTable = new int[UNIGRAM_TABLE_SIZE];
    }
    catch (bad_alloc &ba) {
        cout << "Memory allocation failed: " << ba.what() << endl;
        exit(1);
    }

    int i = 0;
    const real POWER = 0.75f;
    double train_words_pow = 0.0f;
    #pragma omp parallel for num_threads(num_threads) reduction(+: train_words_pow)
    for (int i = 0; i < vocab_size; i++) train_words_pow += pow(vocab[i].cn, POWER);
    double d1 = pow(vocab[i].cn, POWER) / train_words_pow;
    i = 1;
    for (int a = 0; a < UNIGRAM_TABLE_SIZE; a++) {
        unigramTable[a] = i;
        if (a / (double) UNIGRAM_TABLE_SIZE > d1) {
            if (i < vocab_size - 1) i++;
            d1 += pow(vocab[i].cn, POWER) / train_words_pow;
        }
    }

    // Table used for subsampling probability distribution
    if (sample > 0) {
        // Allocate
        try {
            subsampling_probability = new real[vocab_size];
        }
        catch (bad_alloc &ba) {
            cout << "Failed to allocate subsampling probability table: " << ba.what() << endl;
            exit(1);
        }

        real base_probability = sample * train_words;
        // Calculate each probability once (unchanged throughout runtime)
        #pragma omp parallel for num_threads(num_threads) schedule(static, 1)
        for(int i = 0; i < vocab_size; i++) {
            real ratio = base_probability / vocab[i].cn;
            subsampling_probability[i] = sqrtf(ratio) + ratio;
        }
    }

    // Ensure all CUDA copies completed successfully before continuing
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    if (debug_mode > 0) fprintf(stderr, "Network ready\n");
}

// mmap() version of ReadWord()
void ReadWord(char *word, const char *mmap, const u_longlong max_map, u_longlong *cur_map) {
    int wordLen = 0;
    char read = 0;
    u_longlong pos = *cur_map;
    while (pos < max_map) {
        read = mmap[pos];
        pos++;
        if (read == 13) continue;
        else if (read == '\n') {
            if (wordLen > 0) {
                pos--;
                break;
            }
            strcpy(word, (char *) "</s>");
            *cur_map = pos;
            return;
        }
        else if (read == ' ' || read == '\t') {
            if (wordLen > 0) break;
            continue;
        }
        word[wordLen] = read;
        // Truncation rule (only increase while under max length)
        wordLen++;
        if (wordLen >= MAX_STRING - 1) break;
    }
    word[wordLen] = 0;
    // Normal exit
    if (pos == max_map || read == ' ' || read == '\t' || read == '\n') {
        *cur_map = pos;
        return;
    }
    // Truncation
    while (pos < max_map && (read != ' ' && read != '\t' && read != '\n')) {
        read = mmap[pos];
        pos++;
    }
    *cur_map = pos - 1;
}

// Reads next word and returns its index in the vocabulary
int ReadWordIndex(char *word, const char *mapped, u_longlong *pos) {
    int index = -1;
    while(*pos < file_size && index == -1) {
        ReadWord(word, mapped, file_size, pos);
        index = SearchVocab(word);
    }
    return index;
}

void LearnVocabFromTrainFile(const char *mapped) {
    char word[MAX_STRING];
    train_words = 0;
    vocab_size = 0;

    if (debug_mode > 0) fprintf(stderr, "Determining vocabulary present in training data\n");

    AddWordToVocab((char *) "</s>");
    for(u_longlong map_pos = 0; map_pos < file_size; ) {
      ReadWord(word, mapped, file_size, &map_pos);
      if (map_pos == file_size) break;
      train_words++;
      if ((debug_mode > 1) && (train_words % 100000 == 0)) {
          fprintf(stderr, "%lldK%c", train_words / 1000, 13);
          fflush(stderr);
      }
      int i = SearchVocab(word);
      if (i == -1) {
          int a = AddWordToVocab(word);
          vocab[a].cn = 1;
      }
      else vocab[i].cn++;
      if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
    }

    SortVocab();
    train_words++;
    if (debug_mode > 0) fprintf(stderr, "Vocab size: %d\nWords in train file: %lld\n", vocab_size, train_words);
}

// Updated to use mmap()
void ReadVocab() {
    char word[MAX_STRING];
    const char *mapped = NULL;
    u_longlong size;
    {
        int fd = open(read_vocab_file, O_RDONLY);
        struct stat fd_stats;
        if (fstat(fd, &fd_stats) < 0) {
            printf("Vocabulary file \"%s\" not found!\n", read_vocab_file);
            exit(1);
        }
        size = fd_stats.st_size;
        mapped = (char *)mmap(0, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
    }

    if (debug_mode > 0) fprintf(stderr, "Loading vocabulary from file: %s\n", read_vocab_file);

    vocab_size = 0;
    // ++ on for loop to skip newline after associated number
    for (u_longlong map_pos = 0; map_pos < size; map_pos++) {
        ReadWord(word, mapped, size, &map_pos);
        int i = AddWordToVocab(word);
        ReadWord(word, mapped, size, &map_pos);
        vocab[i].cn = atoi(word);
    }
    munmap((void *)mapped, size);
    SortVocab();
    if (debug_mode > 0) fprintf(stderr, "Vocab size: %d\nWords in train file: %lld\n", vocab_size, train_words);
}

// Used to transpose vocabulary definitions from one file to another (ie: bigcorpus.read_vocab(littlecorpus.vocab), so counts must be corrected)
void recountVocab(const char *mapped) {
    char word[MAX_STRING];
    if (debug_mode > 0) fprintf(stderr, "Recalibrating vocabulary using training data\n");

    // Turn all vocab count entries to zero (except /s, which should be 1)
    train_words = 0;
    for(int i = 0; i < vocab_size; i++) vocab[i].cn = 0;

    for (u_longlong map_pos = 0; map_pos < file_size; ) {
        ReadWord(word, mapped, file_size, &map_pos);
        if (debug_mode > 0 && train_words / 100000 == 0) {
            fprintf(stderr, "%lldK%c", train_words/ 1000, 13);
            fflush(stderr);
        }
        int i = SearchVocab(word);
        if (i == -1) continue;
        else {
            vocab[i].cn++;
            train_words++;
        }
    }
    // Removes items below min_count and reorders for proper following execution
    SortVocab();
    if (debug_mode > 0) fprintf(stderr, "Vocab size: %d\nWords in train file: %lld\n", vocab_size, train_words);
}

void saveModel() {
    // Retrieve model from GPU
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy2D(Wih, layer_size * sizeof(real), c_syn0, d_pitch * sizeof(real), layer_size * sizeof(real), vocab_size, cudaMemcpyDeviceToHost), "syn0->Wih");

    if (debug_mode > 0) fprintf(stderr, "Saving model to: %s\n", output_file);

    // Save the word vectors
    FILE *fo = fopen(output_file, "wb");
    fprintf(fo, "%d %d\n", vocab_size, layer_size);
    for (int a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab[a].word);
        if (binary) for (int b = 0; b < layer_size; b++) fwrite(&Wih[a * layer_size + b], sizeof(real), 1, fo);
        else for (int b = 0; b < layer_size; b++) fprintf(fo, "%f ", Wih[a * layer_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

u_longlong createGlobalStream(u_longlong max_sent_count, const char *mapped) {
    u_longlong word_count = 0, sent_length = 0, sent_count = 0;
    char word[MAX_STRING];
    int w;
    for (u_longlong pos = 0; pos < file_size; ) {
        if (debug_mode > 0 && pos % 1000 == 0) printf("Caching corpus: %.2lf%% %c", 100*pos/(double)file_size, 13);
        w = ReadWordIndex(word, mapped, &pos);
        // DO NOT INCLUDE w = 0 IN CACHE UNLESS -cache-eos
        if (cache_eos || (!cache_eos && w != 0)) {
            global_stream[word_count] = w;
            if (sent_length == 0) {
                // Record start of new sentence
                global_sentences[sent_count] = &global_stream[word_count];
                sent_count++;
                // Protection from exceeding memory allocation bounds
                if (sent_count >= max_sent_count) {
                    max_sent_count += 1000;
                    if ((global_sentences = (int **) realloc(global_sentences, max_sent_count * sizeof(int *))) == NULL) {
                        printf("Failed to reallocate corpus cache\n");
                        exit(1);
                    }
                }
            }
            word_count++;
            sent_length++;
            // Identify last word of sentence
            if (w == 0 || sent_length >= (u_longlong)MAX_SENTENCE_LENGTH) sent_length = 0;
        }
    }
    if (debug_mode > 0) printf("\n");
    global_stream[word_count] = 0; // ALWAYS set the last word as "</s>"
    // Tighten memory allocation as needed
    if ((global_sentences = (int **) realloc(global_sentences, sent_count * sizeof(int *))) == NULL) {
        printf("Failed to reallocate corpus cache\n");
        exit(1);
    }
    return sent_count;
}

void shuffleSentences(void) {
    if (debug_mode > 0) printf("Shuffling sentences\n");
    for(u_longlong i = sentence_count - 1; i > 0; i--) {
        int j = (int)((rand()/(double)RAND_MAX)*(i+1));
        int *swp = global_sentences[j];
        global_sentences[j] = global_sentences[i];
        global_sentences[i] = swp;
    }
}

void saveCorpus() {
    if (debug_mode > 0) fprintf(stderr, "Saving corpus replication to: %s\n", save_corpus_file);

    FILE *fo = fopen(save_corpus_file, "w");
    for (u_int s = 0, w; s < sentence_count; s++) {
        for (w = 0; w < (u_int)MAX_SENTENCE_LENGTH && global_sentences[s][w] != 0; w++) fprintf(fo, "%s ", vocab[global_sentences[s][w]].word);
        if (w > 0) fprintf(fo, "\n");
        if (debug_mode > 0) printf("Saving %.2lf%%%c", (100*s)/(double)sentence_count, 13);
    }
    if (debug_mode > 0) printf("\n");
    fclose(fo);
}

void saveCorpusCache() {
    if (debug_mode > 0) fprintf(stderr, "Saving corpus cache to: %s\n", save_corpus_cache_file);

    FILE *fo = fopen(save_corpus_cache_file, "w");
    fprintf(fo, "%llu %llu\n", sentence_count, train_words);
    for (u_int s = 0, w; s < sentence_count; s++) {
        for (w = 0; w < (u_int)MAX_SENTENCE_LENGTH && global_sentences[s][w] != 0; w++) fprintf(fo, "%u ", global_sentences[s][w]);
        if (w > 0) fprintf(fo, "\n");
        if (debug_mode > 0) printf("Saving: %.2lf%%%c", (100*s)/(double)sentence_count, 13);
    }
    if (debug_mode > 0) printf("\n");
    fclose(fo);
}

void readCorpusCache() {
    char word[MAX_STRING];
    const char *mapped = NULL;
    u_longlong size, map_pos = 0;
    {
        int fd = open(read_corpus_cache_file, O_RDONLY);
        struct stat fd_stats;
        if (fstat(fd, &fd_stats) < 0) {
            printf("Corpus cache file \"%s\" not found!\n", read_corpus_cache_file);
            exit(1);
        }
        size = fd_stats.st_size;
        mapped = (char *) mmap(0, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
    }

    if (debug_mode > 0) fprintf(stderr, "Loading corpus cache from file: %s\n", read_corpus_cache_file);

    // Get sentence_count, #train_words
    ReadWord(word, mapped, size, &map_pos);
    sentence_count = atoi(word);
    ReadWord(word, mapped, size, &map_pos);
    train_words = atoi(word);
    map_pos++;

    // Sentence count is guaranteed minimum # sentences
    // However, if MSL now doesn't match MSL when the cache was created, could have more!
    // To be precise, every sentence caused by new MSL could be new
    // Add that many positions into the stream to ensure it is big enough
    u_longlong max_sentences = sentence_count + (train_words / MAX_SENTENCE_LENGTH) + (train_words % MAX_SENTENCE_LENGTH != 0);
    global_sentences = (int **) malloc(max_sentences * sizeof(int *));
    try {
        global_stream = new int[train_words + max_sentences + 1];
    }
    catch (bad_alloc &ba) {
        cout << "Memory allocation failed: " << ba.what() << endl;
        exit(1);
    }

    u_longlong sentence_pos = 0, stream_pos = 0;
    global_sentences[sentence_pos] = &global_stream[0];
    sentence_pos++;
    for (u_longlong msl_trigger = 0; map_pos < size; ) {
        ReadWord(word, mapped, size, &map_pos);
        if (!strcmp(word, (char*)"</s>")) {
            if (cache_eos) {
                global_stream[stream_pos] = 0;
                stream_pos++;
                if (sentence_pos + 1 == max_sentences) {
                    max_sentences *= 2;
                    global_sentences = (int **) realloc(global_sentences, max_sentences * sizeof(int *));
                }
                global_sentences[sentence_pos] = &global_stream[stream_pos];
                sentence_pos++;
                msl_trigger = 0;
            }
        }
        else {
            if (msl_trigger == (u_longlong)MAX_SENTENCE_LENGTH) {
                global_stream[stream_pos] = 0;
                stream_pos++;
                if (sentence_pos + 1 == max_sentences) {
                    max_sentences *= 2;
                    global_sentences = (int **) realloc(global_sentences, max_sentences * sizeof(int *));
                }
                global_sentences[sentence_pos] = &global_stream[stream_pos];
                sentence_pos++;
                msl_trigger = 0;
            }
            else msl_trigger++;
            global_stream[stream_pos] = atoi(word);
            stream_pos++;
        }
        if (debug_mode > 0 && map_pos % 10000 == 0) {
            fprintf(stdout, "Load %.2lf%% %c", (100*map_pos)/(double)size, 13);
            fflush(stdout);
        }
    }
    // Ensure correct sentence count
    sentence_count = sentence_pos - 1;
    if (debug_mode > 0) fprintf(stderr, "\nLoaded %llu word-like items into %llu sentences\n", stream_pos, sentence_pos);
}

void setCommandLineValues(int argc, char **argv) {
    if (argc == 1 || ArgPos((char *) "-help", argc, argv) > 0) {
        printf("Improved word2vec (sgns) in shared memory system with GPU processing\n\n");
        printf("Options:\n");
        printf("** FILES & I/O **\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors\n");
        printf("\t-no-export\n");
        printf("\t\tSkips exporting vectors to file after training (useful for debugging)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-recount\n");
        printf("\t\tReconstruct vocabulary counts from training data\n");
        printf("\t-save-corpus <file>\n");
        printf("\t\tThe corpus as interpreted will be saved to <file>\n");
        printf("\t-save-corpus-cache <file>\n");
        printf("\t\tThe corpus cache will be saved to <file>\n");
        printf("\t-read-corpus-cache <file>\n");
        printf("\t\tThe corpus cache will be read from <file>, not constructed from the training data\n");
        printf("\t-cache-eos\n");
        printf("\t\tInclude cached EOS tokens from corpus paragraph delimiters\n");
        #ifdef SHUFFLE
        printf("\t-pre-shuffle\n");
        printf("\t\tShuffle sentences before first epoch\n");
        printf("\t-epoch-shuffle\n");
        printf("\t\tShuffle sentences at the end of each epoch\n");
        #endif

        printf("** TRAINING PARAMETERS **\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-epoch(s), -iter(s) <int>\n");
        printf("\t\tNumber of training epochs (default 5)\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-negative(s) <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");

        printf("** PARALLEL UTILIZATION **\n");
        printf("\t-thread(s) <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-kernel-batch-size <int>\n");
        printf("\t\tThe batch size used for mini-batch training; default is 200\n");
        printf("\t-streams <int>\n");
        printf("\t\tThe number of CUDA streams each CPU thread manages; default is 4\n");
        printf("\t-max-sentence-length, -msl <int>\n");
        printf("\t\tThe maximum length of a single sentence during training\n");

        printf("\nExamples:\n");
        printf("./%s -train data.txt -output vec.txt -size 128 -window 5 -sample 1e-4 -negative 5 -epoch 3\n\n", argv[0]);
        exit(0);
    }

    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;

    int i;
    if ((i = ArgPos((char *) "-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-no-export", argc, argv)) > 0) _export = false;
    if ((i = ArgPos((char *) "-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-recount", argc, argv)) > 0) recount = true;
    if ((i = ArgPos((char *) "-save-corpus", argc, argv)) > 0) strcpy(save_corpus_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-save-corpus-cache", argc, argv)) > 0) strcpy(save_corpus_cache_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-read-corpus-cache", argc, argv)) > 0) strcpy(read_corpus_cache_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-cache-eos", argc, argv)) > 0) cache_eos = true;
    if ((i = ArgPos((char *) "-pre-shuffle", argc, argv)) > 0) pre_shuffle = true;
    if ((i = ArgPos((char *) "-epoch-shuffle", argc, argv)) > 0) epoch_shuffle = true;

    if ((i = ArgPos((char *) "-size", argc, argv)) > 0) layer_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-epoch", argc, argv)) > 0 || (i = ArgPos((char *)"-epochs", argc, argv)) > 0 || (i = ArgPos((char *)"-iter", argc, argv)) > 0 || (i = ArgPos((char *)"-iters", argc, argv)) > 0) epoch = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-negative", argc, argv)) > 0 || (i = ArgPos((char *)"-negatives", argc, argv)) > 0) negative = atoi(argv[i + 1]);

    if ((i = ArgPos((char *) "-thread", argc, argv)) > 0 || (i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-kernel-batch-size", argc, argv)) > 0) kernel_batch_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-streams", argc, argv)) > 0) streams_per_thread = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-msl", argc, argv)) > 0 || (i = ArgPos((char *) "-max-sentence-length", argc, argv)) > 0) MAX_SENTENCE_LENGTH = atoi(argv[i + 1]);

    if(debug_mode > 0) {
        fprintf(stderr, "** FILES & I/O **\n");
        if (!_export && output_file[0] != 0) fprintf(stderr, "!!\tProcess will train but will not export a model!\n");
        else if (output_file[0] != 0) fprintf(stderr, "Trained model will export to file: %s\n", output_file);
        else fprintf(stderr, "!!\tPreprocessing only -- no training.\n");
        if (read_corpus_cache_file[0] != 0) fprintf(stderr, "Corpus (serialized cache): %s\n", read_corpus_cache_file);
        else fprintf(stderr, "Corpus (file): %s\n", train_file);
        if (cache_eos) fprintf(stderr, "Sentences terminated by corpus paragraphs or maximum of %d words\n", MAX_SENTENCE_LENGTH);
        else fprintf(stderr, "Sentences ONLY terminated by maximum of %d words\n", MAX_SENTENCE_LENGTH);
        #ifdef SHUFFLE
        if (pre_shuffle) fprintf(stderr, "There will be an initial shuffle before the first epoch\n");
        else fprintf(stderr, "The first epoch will use given sentence order\n");
        if (epoch_shuffle) fprintf(stderr, "Sentences are shuffled at the end of each epoch\n");
        #endif

        fprintf(stderr, "\n** TRAINING PARAMETERS **\n");
        fprintf(stderr, "layer size: %d\n", layer_size);
        fprintf(stderr, "train for %d epochs\n", epoch);
        fprintf(stderr, "initial learning rate: %.5f\n", alpha);
        fprintf(stderr, "window size: %d\n", window);
        fprintf(stderr, "subsampling rate: %lf\n", sample);
        fprintf(stderr, "number of negative samples: %d\n", negative);

        fprintf(stderr, "\n** PARALLEL UTILIZATION **\n");
        fprintf(stderr, "#CPU threads: %d\n", num_threads);
        fprintf(stderr, "GPU kernel batch size: %d\n", kernel_batch_size);
        fprintf(stderr, "GPU streams per CPU thread: %d\n", streams_per_thread);

        fprintf(stderr, "\n** BONUS DEBUG **\n");
        fprintf(stderr, "Low Word Count: %d\n", low_word_count);
        fprintf(stderr, "\n");
    }
}

