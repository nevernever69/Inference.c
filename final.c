#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "Tokenize.h"
#include <stdbool.h>
#include "timer.h"

typedef struct {
    char name[64];
    float *data;
    int ndim;
    int dims[4];
    size_t size;
} Tensor;

// Structure to help with detokenization
typedef struct {
    int id;
    char *token;
} TokenEntry;

// Improved KV cache structure
typedef struct {
    float *k_cache; // Shape: [num_layers, max_seq_len, d_model] - flattened
    float *v_cache; // Shape: [num_layers, max_seq_len, d_model] - flattened
    int max_seq_len;
    int current_len;
    int num_layers;
    int d_model;
} KVCache;

// initialize KV cache
KVCache* initialize_kv_cache(int max_seq_len, int num_layers, int d_model) {
    KVCache *cache = malloc(sizeof(KVCache));
    if (!cache) {
        fprintf(stderr, "Memory allocation failed for KVCache\n");
        exit(1);
    }

    cache->max_seq_len = max_seq_len;
    cache->current_len = 0;
    cache->num_layers = num_layers;
    cache->d_model = d_model;
    
    // Allocate memory for K and V caches in one contiguous block per cache
    size_t cache_size = (size_t)num_layers * max_seq_len * d_model * sizeof(float);
    cache->k_cache = malloc(cache_size);
    cache->v_cache = malloc(cache_size);
    
    if (!cache->k_cache || !cache->v_cache) {
        fprintf(stderr, "Memory allocation failed for KV cache arrays\n");
        if (cache->k_cache) free(cache->k_cache);
        if (cache->v_cache) free(cache->v_cache);
        free(cache);
        exit(1);
    }

    // Initialize with zeros
    memset(cache->k_cache, 0, cache_size);
    memset(cache->v_cache, 0, cache_size);

    return cache;
}

// function to free KV cache
void free_kv_cache(KVCache *cache) {
    if (!cache) return;
    free(cache->k_cache);
    free(cache->v_cache);
    free(cache);
}

// function to get pointer to specific position in K cache
float* get_k_cache_ptr(KVCache *cache, int layer_idx, int pos_idx) {
    return &cache->k_cache[(layer_idx * cache->max_seq_len + pos_idx) * cache->d_model];
}

// function to get pointer to specific position in V cache
float* get_v_cache_ptr(KVCache *cache, int layer_idx, int pos_idx) {
    return &cache->v_cache[(layer_idx * cache->max_seq_len + pos_idx) * cache->d_model];
}



Tensor *load_gpt2_model(const char *filename, int *num_tensors_out) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen"); exit(1); }

    Tensor *tensors = NULL;
    int num_tensors = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        Tensor t = {0};
        strncpy(t.name, line, sizeof(t.name)-1);
        size_t len = strlen(t.name);
        if (t.name[len-1] == '\n') t.name[len-1] = '\0';

        if (!fgets(line, sizeof(line), f)) { fprintf(stderr, "Unexpected EOF\n"); exit(1); }
        sscanf(line, "%d %d %d %d %d", &t.ndim, &t.dims[0], &t.dims[1], &t.dims[2], &t.dims[3]);
        t.size = 1;
        for (int i = 0; i < t.ndim; i++) t.size *= t.dims[i];

        t.data = malloc(t.size * sizeof(float));
        fread(t.data, sizeof(float), t.size, f);

        tensors = realloc(tensors, (num_tensors + 1) * sizeof(Tensor));
        tensors[num_tensors++] = t;
    }

    fclose(f);
    *num_tensors_out = num_tensors;
    return tensors;
}

void print_tensor_info(Tensor *t) {
    printf("Tensor %s: shape [", t->name);
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->dims[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("], size=%zu floats\n", t->size);
}

Tensor *find_tensor_by_name(Tensor *tensors, int num_tensors, const char *name) {
    for (int i = 0; i < num_tensors; i++) {
        if (strcmp(tensors[i].name, name) == 0) {
            return &tensors[i];
        }
    }
    return NULL;
}

void layer_norm(float *input, float *output, float *gamma, float *beta, int N, int d, float epsilon) {
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            sum += input[n * d + i];
        }
        float mean = sum / d;

        float sum_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            float diff = input[n * d + i] - mean;
            sum_sq += diff * diff;
        }
        float variance = sum_sq / d;
        float std = sqrtf(variance + epsilon);

        for (int i = 0; i < d; i++) {
            output[n * d + i] = ((input[n * d + i] - mean) / std) * gamma[i] + beta[i];
        }
    }
}

void matmul(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int p = 0; p < n; p++) {
                C[i * k + j] += A[i * n + p] * B[p * k + j];
            }
        }
    }
}

// self-attention function to use KV cache
void self_attention_with_cache(float *input, float *output,
                             float *c_attn_weight, float *c_attn_bias,
                             float *c_proj_weight, float *c_proj_bias,
                             int seq_len, int d_model, int num_heads, int d_k,
                             KVCache *cache, int layer_idx, bool use_cache,
                             int position_id) {
    // using the cache, we only need to compute for the new token
    int process_len = use_cache ? 1 : seq_len;
    int start_pos = use_cache ? seq_len - 1 : 0;
    
    // Compute QKV projections
    float *qkv = malloc(process_len * 3 * d_model * sizeof(float));
    if (!qkv) {
        fprintf(stderr, "Memory allocation failed for QKV\n");
        exit(1);
    }

    // QKV projection for current token(s)

    // Note: c_attn_weight should be of shape [d_model, 3 * d_model]
    // and it will be mulliplied with input of shape [process_len, d_model] 
    // like qkv = input * c_attn_weight
    matmul(&input[start_pos * d_model], c_attn_weight, qkv, process_len, d_model, 3 * d_model);
    for (int i = 0; i < process_len; i++) {
        for (int j = 0; j < 3 * d_model; j++) {
            qkv[i * (3 * d_model) + j] += c_attn_bias[j];
        }
    }

    // Split QKV into separate arrays
    // create Q, K, V arrays to store the split values
    float *Q = malloc(process_len * d_model * sizeof(float));
    float *K = malloc(process_len * d_model * sizeof(float));
    float *V = malloc(process_len * d_model * sizeof(float));
    if (!Q || !K || !V) {
        fprintf(stderr, "Memory allocation failed for Q, K, V\n");
        free(qkv);
        if (Q) free(Q);
        if (K) free(K);
        if (V) free(V);
        exit(1);
    }
    // Split QKV into Q, K, V
    for (int i = 0; i < process_len; i++) {
        for (int j = 0; j < d_model; j++) {
            Q[i * d_model + j] = qkv[i * (3 * d_model) + j];
            K[i * d_model + j] = qkv[i * (3 * d_model) + d_model + j];
            V[i * d_model + j] = qkv[i * (3 * d_model) + 2 * d_model + j];
        }
    }

    // Store K, V in cache
    if (use_cache) {
        // Store the current token's K, V values in the cache at position_id
	printf("%d\n", d_model);
        memcpy(get_k_cache_ptr(cache, layer_idx, position_id), K, d_model * sizeof(float));
        memcpy(get_v_cache_ptr(cache, layer_idx, position_id), V, d_model * sizeof(float));
        cache->current_len = position_id + 1;
    } else {
        // Populate cache with all tokens in initial pass
        for (int pos = 0; pos < seq_len; pos++) {
            memcpy(get_k_cache_ptr(cache, layer_idx, pos), &K[pos * d_model], d_model * sizeof(float));
            memcpy(get_v_cache_ptr(cache, layer_idx, pos), &V[pos * d_model], d_model * sizeof(float));
        }
        cache->current_len = seq_len;
    }
    

    // Output for attention result
    float *attn_output = calloc(seq_len * d_model, sizeof(float));
    if (!attn_output) {
        fprintf(stderr, "Memory allocation failed for attention output\n");
        free(qkv); free(Q); free(K); free(V);
        exit(1);
    }

    // Perform attention for each head
    for (int h = 0; h < num_heads; h++) {
        int offset = h * d_k;
        
        // Process each position in the sequence that we need to compute
        for (int i = start_pos; i < seq_len; i++) {
            // Calculate attention scores with all previous positions up to i
            int context_len = use_cache ? cache->current_len : (i + 1);
            float *scores = malloc(context_len * sizeof(float));
            if (!scores) {
                fprintf(stderr, "Memory allocation failed for scores\n");
                free(qkv); free(Q); free(K); free(V); free(attn_output);
                exit(1);
            }

            float max_score = -FLT_MAX;
            
            // Get current Q vector for this head
            float *q_ptr = &Q[(i - start_pos) * d_model + offset];
            
            // Calculate attention scores with all context tokens
            for (int j = 0; j < context_len; j++) {
                scores[j] = 0.0f;
                float *k_ptr;
                
                // Use cached K values
                if (use_cache) {
                    k_ptr = get_k_cache_ptr(cache, layer_idx, j) + offset;
                } else {
                    // For initial pass, use computed K values
                    k_ptr = &K[j * d_model + offset];
                }
                
                // Compute dot product
                for (int k = 0; k < d_k; k++) {
                    scores[j] += q_ptr[k] * k_ptr[k];
                }
                
                // Scale by sqrt(d_k)
                scores[j] /= sqrtf((float)d_k);
                if (scores[j] > max_score) {
                    max_score = scores[j];
                }
            }

            // Softmax: Convert scores to probabilities
            float sum_exp = 0.0f;
            for (int j = 0; j < context_len; j++) {
                scores[j] = expf(scores[j] - max_score); // Subtract max for numerical stability
                sum_exp += scores[j];
            }
            
            // Normalize
            for (int j = 0; j < context_len; j++) {
                scores[j] /= sum_exp;
            }

            // Apply attention weights to values
            for (int k = 0; k < d_k; k++) {
                float weighted_sum = 0.0f;
                for (int j = 0; j < context_len; j++) {
                    float v_value;
                    // Use cached V values
                    if (use_cache) {
                        v_value = get_v_cache_ptr(cache, layer_idx, j)[offset + k];
                    } else {
                        // For initial pass, use computed V values
                        v_value = V[j * d_model + offset + k];
                    }
                    weighted_sum += scores[j] * v_value;
                }
                attn_output[i * d_model + offset + k] = weighted_sum;
            }
            
            free(scores);
        }
    }

    // Final projection
    float *proj_output = malloc(seq_len * d_model * sizeof(float));
    if (!proj_output) {
        fprintf(stderr, "Memory allocation failed for proj output\n");
        free(qkv); free(Q); free(K); free(V); free(attn_output);
        exit(1);
    }

    // If using cache, only compute projection for the new token
    if (use_cache) {
        matmul(&attn_output[(seq_len-1) * d_model], c_proj_weight, &proj_output[(seq_len-1) * d_model], 1, d_model, d_model);
        for (int j = 0; j < d_model; j++) {
            output[(seq_len-1) * d_model + j] = proj_output[(seq_len-1) * d_model + j] + c_proj_bias[j];
        }
    } else {
        matmul(attn_output, c_proj_weight, proj_output, seq_len, d_model, d_model);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_model; j++) {
                output[i * d_model + j] = proj_output[i * d_model + j] + c_proj_bias[j];
            }
        }
    }

    // Clean up
    free(qkv);
    free(Q);
    free(K);
    free(V);
    free(attn_output);
    free(proj_output);
}

void gelu(float *input, float *output, int size) {
    const float sqrt_2_pi = 0.7978845608;
    const float c = 0.044715;
    for (int i = 0; i < size; i++) {
        float x = input[i];
        float x3 = x * x * x;
        float tanh_arg = sqrt_2_pi * (x + c * x3);
        output[i] = 0.5 * x * (1.0 + tanhf(tanh_arg));
    }
}

// Feed forward for single token (optimization for cached inference)
void feed_forward_single_token(float *input, float *output, 
                            float *c_fc_weight, float *c_fc_bias,
                            float *c_proj_weight, float *c_proj_bias, 
                            int d_model, int d_ff) {
    // Process single token through feed-forward network
    float *fc_output = malloc(d_ff * sizeof(float));
    float *gelu_output = malloc(d_ff * sizeof(float));
    
    if (!fc_output || !gelu_output) {
        fprintf(stderr, "Memory allocation failed for feed-forward\n");
        exit(1);
    }
    
    // FC layer
    matmul(input, c_fc_weight, fc_output, 1, d_model, d_ff);
    for (int j = 0; j < d_ff; j++) {
        fc_output[j] += c_fc_bias[j];
    }
    
    // GELU activation
    gelu(fc_output, gelu_output, d_ff);
    
    // Projection
    matmul(gelu_output, c_proj_weight, output, 1, d_ff, d_model);
    for (int j = 0; j < d_model; j++) {
        output[j] += c_proj_bias[j];
    }
    
    free(fc_output);
    free(gelu_output);
}

// Standard feed-forward function for full sequence
void feed_forward(float *input, float *output, float *c_fc_weight, float *c_fc_bias,
                float *c_proj_weight, float *c_proj_bias, int seq_len, int d_model, int d_ff) {
    float *fc_output = malloc(seq_len * d_ff * sizeof(float));
    float *gelu_output = malloc(seq_len * d_ff * sizeof(float));
    
    if (!fc_output || !gelu_output) {
        fprintf(stderr, "Memory allocation failed for feed-forward\n");
        exit(1);
    }
    
    matmul(input, c_fc_weight, fc_output, seq_len, d_model, d_ff);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_ff; j++) {
            fc_output[i * d_ff + j] += c_fc_bias[j];
        }
    }
    
    gelu(fc_output, gelu_output, seq_len * d_ff);
    
    matmul(gelu_output, c_proj_weight, output, seq_len, d_ff, d_model);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            output[i * d_model + j] += c_proj_bias[j];
        }
    }
    
    free(fc_output);
    free(gelu_output);
}

// Modified transformer block to use KV cache
void transformer_block_with_cache(float *input, float *output, 
                               Tensor *tensors, int num_tensors,
                               int layer_idx, int seq_len, int d_model, 
                               int num_heads, int d_k, int d_ff, float epsilon,
                               KVCache *cache, bool use_cache,
                               int position_id) {
    
    char ln_1_weight_name[64], ln_1_bias_name[64];
    char ln_2_weight_name[64], ln_2_bias_name[64];
    char c_attn_weight_name[64], c_attn_bias_name[64];
    char c_proj_weight_name[64], c_proj_bias_name[64];
    char c_fc_weight_name[64], c_fc_bias_name[64];
    char c_proj_mlp_weight_name[64], c_proj_mlp_bias_name[64];

    snprintf(ln_1_weight_name, sizeof(ln_1_weight_name), "h.%d.ln_1.weight", layer_idx);
    snprintf(ln_1_bias_name, sizeof(ln_1_bias_name), "h.%d.ln_1.bias", layer_idx);
    snprintf(ln_2_weight_name, sizeof(ln_2_weight_name), "h.%d.ln_2.weight", layer_idx);
    snprintf(ln_2_bias_name, sizeof(ln_2_bias_name), "h.%d.ln_2.bias", layer_idx);
    snprintf(c_attn_weight_name, sizeof(c_attn_weight_name), "h.%d.attn.c_attn.weight", layer_idx);
    snprintf(c_attn_bias_name, sizeof(c_attn_bias_name), "h.%d.attn.c_attn.bias", layer_idx);
    snprintf(c_proj_weight_name, sizeof(c_proj_weight_name), "h.%d.attn.c_proj.weight", layer_idx);
    snprintf(c_proj_bias_name, sizeof(c_proj_bias_name), "h.%d.attn.c_proj.bias", layer_idx);
    snprintf(c_fc_weight_name, sizeof(c_fc_weight_name), "h.%d.mlp.c_fc.weight", layer_idx);
    snprintf(c_fc_bias_name, sizeof(c_fc_bias_name), "h.%d.mlp.c_fc.bias", layer_idx);
    snprintf(c_proj_mlp_weight_name, sizeof(c_proj_mlp_weight_name), "h.%d.mlp.c_proj.weight", layer_idx);
    snprintf(c_proj_mlp_bias_name, sizeof(c_proj_mlp_bias_name), "h.%d.mlp.c_proj.bias", layer_idx);

    Tensor *ln_1_weight = find_tensor_by_name(tensors, num_tensors, ln_1_weight_name);
    Tensor *ln_1_bias = find_tensor_by_name(tensors, num_tensors, ln_1_bias_name);
    Tensor *c_attn_weight = find_tensor_by_name(tensors, num_tensors, c_attn_weight_name);
    Tensor *c_attn_bias = find_tensor_by_name(tensors, num_tensors, c_attn_bias_name);
    Tensor *c_proj_weight = find_tensor_by_name(tensors, num_tensors, c_proj_weight_name);
    Tensor *c_proj_bias = find_tensor_by_name(tensors, num_tensors, c_proj_bias_name);
    Tensor *c_fc_weight = find_tensor_by_name(tensors, num_tensors, c_fc_weight_name);
    Tensor *c_fc_bias = find_tensor_by_name(tensors, num_tensors, c_fc_bias_name);
    Tensor *c_proj_mlp_weight = find_tensor_by_name(tensors, num_tensors, c_proj_mlp_weight_name);
    Tensor *c_proj_mlp_bias = find_tensor_by_name(tensors, num_tensors, c_proj_mlp_bias_name);
    Tensor *ln_2_weight = find_tensor_by_name(tensors, num_tensors, ln_2_weight_name);
    Tensor *ln_2_bias = find_tensor_by_name(tensors, num_tensors, ln_2_bias_name);

    if (!ln_1_weight || !ln_1_bias || !c_attn_weight || !c_attn_bias ||
        !c_proj_weight || !c_proj_bias || !c_fc_weight || !c_fc_bias ||
        !c_proj_mlp_weight || !c_proj_mlp_bias || !ln_2_weight || !ln_2_bias) {
        fprintf(stderr, "ERROR: Missing required tensors for layer %d\n", layer_idx);
        exit(1);
    }
    
    // Process differently depending on whether we're using cache
    if (use_cache) {
        // When using cache, we only need to process the new token
        float *ln_1_output = malloc(d_model * sizeof(float));
        float *attn_output = malloc(seq_len * d_model * sizeof(float));
        float *residual_1 = malloc(d_model * sizeof(float));
        float *ln_2_output = malloc(d_model * sizeof(float));
        float *ffn_output = malloc(d_model * sizeof(float));
        
        if (!ln_1_output || !attn_output || !residual_1 || !ln_2_output || !ffn_output) {
            fprintf(stderr, "Memory allocation failed in transformer_block\n");
            exit(1);
        }
        
        // Apply layer norm to current token only
        layer_norm(&input[(seq_len-1) * d_model], ln_1_output, ln_1_weight->data, ln_1_bias->data, 1, d_model, epsilon);
        
        // Copy ln_1_output to full input for attention
        float *attn_input = malloc(seq_len * d_model * sizeof(float));
        if (!attn_input) {
            fprintf(stderr, "Memory allocation failed for attn_input\n");
            exit(1);
        }
        
        // Copy previous tokens' data (not needed for attention computation but keep for output shape)
        memcpy(attn_input, input, (seq_len-1) * d_model * sizeof(float));
        // Copy current token's normalized data
        memcpy(&attn_input[(seq_len-1) * d_model], ln_1_output, d_model * sizeof(float));
        
        // Self-attention with cache - process only the new token but attend to all previous tokens via cache
        self_attention_with_cache(attn_input, attn_output, 
                               c_attn_weight->data, c_attn_bias->data,
                               c_proj_weight->data, c_proj_bias->data, 
                               seq_len, d_model, num_heads, d_k,
                               cache, layer_idx, use_cache, position_id);
        
        // Residual connection for current token only
        for (int j = 0; j < d_model; j++) {
            residual_1[j] = input[(seq_len-1) * d_model + j] + attn_output[(seq_len-1) * d_model + j];
        }
        
        // Second layer norm for current token
        layer_norm(residual_1, ln_2_output, ln_2_weight->data, ln_2_bias->data, 1, d_model, epsilon);
        
        // Feed-forward for current token only
        feed_forward_single_token(ln_2_output, ffn_output, 
                               c_fc_weight->data, c_fc_bias->data,
                               c_proj_mlp_weight->data, c_proj_mlp_bias->data, 
                               d_model, d_ff);
        
        // Final residual connection and copy to output
        for (int j = 0; j < d_model; j++) {
            output[(seq_len-1) * d_model + j] = residual_1[j] + ffn_output[j];
        }
        
        // Copy previous tokens' output (unchanged when using cache)
        if (seq_len > 1) {
            memcpy(output, input, (seq_len-1) * d_model * sizeof(float));
        }
        
        // Clean up
        free(ln_1_output);
        free(attn_input);
        free(attn_output);
        free(residual_1);
        free(ln_2_output);
        free(ffn_output);
    } else {
        // Standard processing for full sequence (initial pass)
        float *ln_1_output = malloc(seq_len * d_model * sizeof(float));
        float *attn_output = malloc(seq_len * d_model * sizeof(float));
        float *residual_1 = malloc(seq_len * d_model * sizeof(float));
        float *ln_2_output = malloc(seq_len * d_model * sizeof(float));
        float *ffn_output = malloc(seq_len * d_model * sizeof(float));
        
        if (!ln_1_output || !attn_output || !residual_1 || !ln_2_output || !ffn_output) {
            fprintf(stderr, "Memory allocation failed in transformer_block\n");
            exit(1);
        }
        
        // First LayerNorm
        layer_norm(input, ln_1_output, ln_1_weight->data, ln_1_bias->data, seq_len, d_model, epsilon);
        
        // Self-Attention with cache population (but not using cached values yet)
        self_attention_with_cache(ln_1_output, attn_output, 
                               c_attn_weight->data, c_attn_bias->data,
                               c_proj_weight->data, c_proj_bias->data, 
                               seq_len, d_model, num_heads, d_k,
                               cache, layer_idx, false, position_id);
        
        // Residual connection
        for (int i = 0; i < seq_len * d_model; i++) {
            residual_1[i] = input[i] + attn_output[i];
        }
        
        // Second LayerNorm
        layer_norm(residual_1, ln_2_output, ln_2_weight->data, ln_2_bias->data, seq_len, d_model, epsilon);
        
        // Feed-Forward
        feed_forward(ln_2_output, ffn_output, 
                   c_fc_weight->data, c_fc_bias->data,
                   c_proj_mlp_weight->data, c_proj_mlp_bias->data, 
                   seq_len, d_model, d_ff);
        
        // Final residual connection
        for (int i = 0; i < seq_len * d_model; i++) {
            output[i] = residual_1[i] + ffn_output[i];
        }
        
        // Clean up
        free(ln_1_output);
        free(attn_output);
        free(residual_1);
        free(ln_2_output);
        free(ffn_output);
    }
}

// Function to find the token with the highest probability
int argmax(float *array, int size) {
    int max_idx = 0;
    float max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// Function to apply softmax to logits
void softmax(float *logits, float *probs, int size) {
    // Find max for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    // Compute exponentials and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum_exp += probs[i];
    }

    // Normalize
    for (int i = 0; i < size; i++) {
        probs[i] /= sum_exp;
    }
}

// Function to load token mapping
TokenEntry *load_token_mapping(const char *filename, int *num_tokens) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Failed to open token mapping file %s\n", filename);
        return NULL;
    }

    int capacity = 1000;
    TokenEntry *tokens = malloc(capacity * sizeof(TokenEntry));
    *num_tokens = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        int id;
        char token[512];
        if (sscanf(line, "%d %511s", &id, token) == 2) {
            if (*num_tokens >= capacity) {
                capacity *= 2;
                tokens = realloc(tokens, capacity * sizeof(TokenEntry));
            }
            tokens[*num_tokens].id = id;
            tokens[*num_tokens].token = strdup(token);
            (*num_tokens)++;
        }
    }
    fclose(f);
    return tokens;
}

// Function to get token string from ID
const char *get_token_str(TokenEntry *tokens, int num_tokens, int token_id) {
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i].id == token_id) {
            return tokens[i].token;
        }
    }
    return "";
}

void printclean(int *initial_tokens, int initial_len, TokenEntry *token_mapping, int num_token_entries) {
    for (int i = 0; i < initial_len; i++) {
        const char *token_str = get_token_str(token_mapping, num_token_entries, initial_tokens[i]);
        // Process each character to replace Ġ/Ċ
        for (int j = 0; token_str[j] != '\0'; ) {
            if ((unsigned char)token_str[j] == 0xC4) {
                if ((unsigned char)token_str[j+1] == 0xA0) { // Ġ -> space
                    putchar(' ');
                    j += 2;
                } else if ((unsigned char)token_str[j+1] == 0x8A) { // Ċ -> newline
                    putchar('\n');
                    j += 2;
                } else {
                    putchar(token_str[j]);
                    j++;
                }
            } else {
                putchar(token_str[j]);
                j++;
            }
        }
    }
    fflush(stdout); // Ensure immediate output
}

// Generate a random number between 0 and 1
float random_uniform() {
    return (float)rand() / (float)RAND_MAX;
}

// Apply temperature to logits
void apply_temperature(float *logits, int size, float temperature) {
    if (temperature <= 0.0f) temperature = 1.0f;
    for (int i = 0; i < size; i++) {
        logits[i] /= temperature;
    }
}

// Sample token from probability distribution
int sample_token(float *probs, int size) {
    float r = random_uniform();
    float cdf = 0.0f;
    for (int i = 0; i < size; i++) {
        cdf += probs[i];
        if (r < cdf) return i;
    }
    return size - 1; // Fallback to last token
}

// Top-k filtering: keep only the top k tokens with highest probability
void top_k_filtering(float *logits, int size, int k) {
    if (k >= size || k <= 0) return; // No filtering needed
    // Find the kth largest logit
    float *logits_copy = malloc(size * sizeof(float));
    memcpy(logits_copy, logits, size * sizeof(float));
    // Simple selection algorithm to find kth value
    for (int i = 0; i < k; i++) {
        float max_val = -FLT_MAX;
        int max_idx = -1;
        for (int j = 0; j < size; j++) {
            if (logits_copy[j] > max_val) {
                max_val = logits_copy[j];
                max_idx = j;
            }
        }
        logits_copy[max_idx] = -FLT_MAX;
    }
    
    float kth_value = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (logits_copy[i] > kth_value) {
            kth_value = logits_copy[i];
        }
    }
    
    // Set all logits below the kth value to negative infinity
    for (int i = 0; i < size; i++) {
        if (logits[i] < kth_value) {
            logits[i] = -FLT_MAX;
        }
    }
    
    free(logits_copy);
}

// Top-p (nucleus) sampling: keep smallest set of tokens whose cumulative probability exceeds p
void top_p_filtering(float *logits, float *probs, int size, float p) {
    if (p >= 1.0f || p <= 0.0f) return; // No filtering needed
    // First compute probabilities
    softmax(logits, probs, size);
    // Create index array
    typedef struct {
        int idx;
        float prob;
    } IdxProb;
    IdxProb *items = malloc(size * sizeof(IdxProb));
    for (int i = 0; i < size; i++) {
        items[i].idx = i;
        items[i].prob = probs[i];
    }
    
    // Sort by probability (descending)
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            if (items[j].prob > items[i].prob) {
                IdxProb temp = items[i];
                items[i] = items[j];
                items[j] = temp;
            }
        }
    }
    
    // Compute cumulative probability and find cutoff
    float cumulative = 0.0f;
    int cutoff_idx = size - 1;
    for (int i = 0; i < size; i++) {
        cumulative += items[i].prob;
        if (cumulative >= p) {
            cutoff_idx = i;
            break;
        }
    }
    
    // Set logits below cutoff to negative infinity
    float cutoff_prob = items[cutoff_idx].prob;
    for (int i = 0; i < size; i++) {
        if (probs[i] < cutoff_prob) {
            logits[i] = -FLT_MAX;
        }
    }
    
    free(items);
    // Recompute probabilities
    softmax(logits, probs, size);
}

// Modified generate_text function to use KV cache
void generate_text_with_cache(Tensor *tensors, int num_tensors, int *initial_tokens, int initial_len,
                          int max_new_tokens, TokenEntry *token_mapping, int num_token_entries) {
    // Initialize random seed
    srand(time(NULL));
    
    // Hyperparameters for sampling
    float temperature = 0.7f; // Lower = more focused, higher = more random
    int top_k = 40;           // Consider only top k tokens
    float top_p = 0.9f;       // Consider tokens with cumulative probability < p
    
    // Load model parameters
    Tensor *wte = find_tensor_by_name(tensors, num_tensors, "wte.weight");
    Tensor *wpe = find_tensor_by_name(tensors, num_tensors, "wpe.weight");
    Tensor *ln_f_weight = find_tensor_by_name(tensors, num_tensors, "ln_f.weight");
    Tensor *ln_f_bias = find_tensor_by_name(tensors, num_tensors, "ln_f.bias");
    
    // Model parameters
    int d_model = wte->dims[1];
    int num_heads = 12;
    int d_k = d_model / num_heads;
    int d_ff = 4 * d_model;
    int vocab_size = wte->dims[0];
    float epsilon = 1e-5;
    
    // Count layers
    int num_layers = 0;
    for (int i = 0; i < num_tensors; i++) {
        if (strstr(tensors[i].name, "h.") == tensors[i].name) {
            int layer_idx;
            if (sscanf(tensors[i].name, "h.%d", &layer_idx) == 1) {
                if (layer_idx + 1 > num_layers) {
                    num_layers = layer_idx + 1;
                }
            }
        }
    }
    
    // Print input tokens
    printf("Input: ");
    printclean(initial_tokens, initial_len, token_mapping, num_token_entries);
    printf("\n\nGenerating...\n\n");
    printclean(initial_tokens, initial_len, token_mapping, num_token_entries);
    
    // Allocate memory for the growing sequence
    int max_seq_len = initial_len + max_new_tokens;
    int *token_sequence = malloc(max_seq_len * sizeof(int));
    memcpy(token_sequence, initial_tokens, initial_len * sizeof(int));
    int current_seq_len = initial_len;
    
    // Initialize KV cache
    KVCache *cache = initialize_kv_cache(max_seq_len, num_layers, d_model);
    
    Timer timer;
    start_timer(&timer);
    
    // First pass: process the entire prompt (prefill phase)
    {
        int seq_len = initial_len;
        
        // Compute embeddings for the prompt
        float *embeddings = malloc(seq_len * d_model * sizeof(float));
        if (!embeddings) {
            fprintf(stderr, "Memory allocation failed for embeddings\n");
            exit(1);
        }
        
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_model; j++) {
                embeddings[i * d_model + j] = wte->data[token_sequence[i] * d_model + j] +
                                             wpe->data[i * d_model + j];
            }
        }
        
        // Forward pass through all layers
        float *layer_input = malloc(seq_len * d_model * sizeof(float));
        float *layer_output = malloc(seq_len * d_model * sizeof(float));
        if (!layer_input || !layer_output) {
            fprintf(stderr, "Memory allocation failed for layer buffers\n");
            exit(1);
        }
        
        // Copy embeddings to layer_input
        memcpy(layer_input, embeddings, seq_len * d_model * sizeof(float));
        
        // Process each layer, populating the KV cache
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
            transformer_block_with_cache(layer_input, layer_output, tensors, num_tensors,
                                      layer_idx, seq_len, d_model, num_heads, d_k, d_ff, epsilon,
                                      cache, false, 0); // last parameter is position_id, starts at 0
            
            // Swap input and output for next layer
            float *temp = layer_input;
            layer_input = layer_output;
            layer_output = temp;
        }
        
        // The final output is now in layer_input due to the swap
        
        // Clean up
        free(embeddings);
        free(layer_input);
        free(layer_output);
    }
    
    // Now generate new tokens one by one using the cache
    for (int iter = 0; iter < max_new_tokens; iter++) {
        int seq_len = current_seq_len;
        int position_id = seq_len - 1; // Current position for positional embedding
        
        // Compute embedding only for the new token
        float *embedding = malloc(d_model * sizeof(float));
        if (!embedding) {
            fprintf(stderr, "Memory allocation failed for embedding\n");
            exit(1);
        }
        
        for (int j = 0; j < d_model; j++) {
            embedding[j] = wte->data[token_sequence[position_id] * d_model + j] +
                          wpe->data[position_id * d_model + j];
        }
        
        // Create input with enough space for the full sequence
        float *layer_input = malloc(seq_len * d_model * sizeof(float));
        float *layer_output = malloc(seq_len * d_model * sizeof(float));
        if (!layer_input || !layer_output) {
            fprintf(stderr, "Memory allocation failed for layer buffers\n");
            exit(1);
        }
        
        // For cached inference, we only need the new token at the end
        memcpy(&layer_input[(seq_len-1) * d_model], embedding, d_model * sizeof(float));
        
        // Process each layer
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
            transformer_block_with_cache(layer_input, layer_output, tensors, num_tensors,
                                      layer_idx, seq_len, d_model, num_heads, d_k, d_ff, epsilon,
                                      cache, true, position_id);
            
            // Swap input and output for next layer
            float *temp = layer_input;
            layer_input = layer_output;
            layer_output = temp;
        }
        
        // The final output for the new token is now in layer_input due to the swap
        
        // Compute logits for the last position only
        float *logits = malloc(vocab_size * sizeof(float));
        float *probs = malloc(vocab_size * sizeof(float));
        float *ln_output = malloc(d_model * sizeof(float));
        if (!logits || !probs || !ln_output) {
            fprintf(stderr, "Memory allocation failed for logits\n");
            exit(1);
        }
        
        // Apply layer norm
        layer_norm(&layer_input[(seq_len-1) * d_model], ln_output,
                   ln_f_weight->data, ln_f_bias->data, 1, d_model, epsilon);
        
        // Compute logits
        for (int i = 0; i < vocab_size; i++) {
            logits[i] = 0.0f;
            for (int j = 0; j < d_model; j++) {
                logits[i] += ln_output[j] * wte->data[i * d_model + j];
            }
        }
        
        // Apply temperature to logits
        apply_temperature(logits, vocab_size, temperature);
        
        // Apply top-k filtering
        top_k_filtering(logits, vocab_size, top_k);
        
        // Apply top-p filtering and convert to probabilities
        top_p_filtering(logits, probs, vocab_size, top_p);
        
        // Sample next token from the filtered distribution
        int next_token = sample_token(probs, vocab_size);
        
        // Add the new token to our sequence
        token_sequence[current_seq_len] = next_token;
        current_seq_len++;
        
        // Print the new token
        const char *token_str = get_token_str(token_mapping, num_token_entries, next_token);
        // Print each character, replacing Ġ and Ċ with space and newline
        for (int i = 0; token_str[i] != '\0'; ) {
            // Check for Ġ (UTF-8: 0xC4 0xA0)
            if ((unsigned char)token_str[i] == 0xC4 && (unsigned char)token_str[i+1] == 0xA0) {
                putchar(' ');
                i += 2;
            }
            // Check for Ċ (UTF-8: 0xC4 0x8A)
            else if ((unsigned char)token_str[i] == 0xC4 && (unsigned char)token_str[i+1] == 0x8A) {
                putchar('\n');
                i += 2;
            }
            else {
                putchar(token_str[i]);
                i++;
            }
        }
        fflush(stdout);
        
        // Clean up for this iteration
        free(embedding);
        free(layer_input);
        free(layer_output);
        free(logits);
        free(probs);
        free(ln_output);
        
        // Stop if we generated end token
        if (next_token == 50256) { // <|endoftext|> token in GPT-2
            break;
        }
    }
    
    stop_timer(&timer);
    
    // Calculate metrics
    double elapsed_sec = get_elapsed_sec(&timer);
    int new_tokens = current_seq_len - initial_len;
    double tokens_per_sec = new_tokens / elapsed_sec;
    
    printf("\n\nGenerated %d tokens in %.2f seconds (%.2f tokens/sec)\n",
           new_tokens, elapsed_sec, tokens_per_sec);
    printf("\n\nGeneration complete.\n");
    
    // Clean up
    free(token_sequence);
    free_kv_cache(cache);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s pytorch_model.bin token_mapping.txt input_text\n", argv[0]);
        return 1;
    }
    
    char *model_file = argv[1];
    char *token_map_file = argv[2];
    char *input = argv[3];
    
    int num_tensors;
    Tensor *tensors = load_gpt2_model(model_file, &num_tensors);
    printf("Loaded %d tensors\n", num_tensors);
    
    // Load token mapping
    int num_token_entries;
    TokenEntry *token_mapping = load_token_mapping(token_map_file, &num_token_entries);
    if (!token_mapping) {
        fprintf(stderr, "Failed to load token mapping\n");
        return 1;
    }
    
    // Tokenize input
    int seq_len;
    int *token_ids = tokenize(input, &seq_len);
    if (!token_ids || seq_len == 0) {
        fprintf(stderr, "ERROR: Tokenization failed or no tokens\n");
        return 1;
    }
    
    printf("Input tokenized to %d tokens\n", seq_len);
    
    // Generate 10 new tokens using cache
    generate_text_with_cache(tensors, num_tensors, token_ids, seq_len, 10, token_mapping, num_token_entries);
    
    // Clean up
    free(token_ids);
    for (int i = 0; i < num_tensors; i++) {
        free(tensors[i].data);
    }
    
    free(tensors);
    
    // Free token mapping
    for (int i = 0; i < num_token_entries; i++) {
        free(token_mapping[i].token);
    }
    
    free(token_mapping);
    
    return 0;
}

