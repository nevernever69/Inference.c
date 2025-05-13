#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "Tokenize.h"
#include<time.h>
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

void self_attention(float *input, float *output, float *c_attn_weight, float *c_attn_bias,
                   float *c_proj_weight, float *c_proj_bias, int seq_len, int d_model,
                   int num_heads, int d_k, float *Q, float *K, float *V) {
    
    // Compute query, key, value projections
    float *qkv = malloc(seq_len * 3 * d_model * sizeof(float));
    float *attn_output = malloc(seq_len * d_model * sizeof(float));
    
    // QKV projection
    matmul(input, c_attn_weight, qkv, seq_len, d_model, 3 * d_model);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < 3 * d_model; j++) {
            qkv[i * (3 * d_model) + j] += c_attn_bias[j];
        }
    }
    
    // Split QKV into separate arrays for ease of use
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            Q[i * d_model + j] = qkv[i * (3 * d_model) + j];
            K[i * d_model + j] = qkv[i * (3 * d_model) + d_model + j];
            V[i * d_model + j] = qkv[i * (3 * d_model) + 2 * d_model + j];
        }
    }
    
    // Initialize output to zero
    memset(attn_output, 0, seq_len * d_model * sizeof(float));
    
    // Perform attention for each head
    for (int h = 0; h < num_heads; h++) {
        int offset = h * d_k;
        
        // For each position in the sequence
        for (int i = 0; i < seq_len; i++) {
            // Compute attention scores
            float *scores = malloc(seq_len * sizeof(float));
            float max_score = -FLT_MAX;
            
            // Calculate raw attention scores (Q·K^T / sqrt(d_k))
            for (int j = 0; j <= i; j++) {  // Causal mask: attend only to past
                scores[j] = 0.0f;
                for (int k = 0; k < d_k; k++) {
                    scores[j] += Q[i * d_model + offset + k] * K[j * d_model + offset + k];
                }
                scores[j] /= sqrtf((float)d_k);  // Scale by sqrt(d_k)
                
                if (scores[j] > max_score) {
                    max_score = scores[j];
                }
            }
            
            // Apply causal mask: set scores for future positions to -inf
            for (int j = i+1; j < seq_len; j++) {
                scores[j] = -FLT_MAX;
            }
            
            // Softmax: Convert scores to probabilities
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                if (scores[j] == -FLT_MAX) {
                    scores[j] = 0.0f;  // exp(-inf) = 0
                } else {
                    scores[j] = expf(scores[j] - max_score);  // Subtract max for numerical stability
                    sum_exp += scores[j];
                }
            }
            
            // Normalize
            if (sum_exp > 0.0f) {
                for (int j = 0; j < seq_len; j++) {
                    scores[j] /= sum_exp;
                }
            }
            
            // Apply attention weights to values
            for (int k = 0; k < d_k; k++) {
                float weighted_sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    weighted_sum += scores[j] * V[j * d_model + offset + k];
                }
                attn_output[i * d_model + offset + k] = weighted_sum;
            }
            free(scores);
        }
    }
    
    // Final projection
    matmul(attn_output, c_proj_weight, output, seq_len, d_model, d_model);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            output[i * d_model + j] += c_proj_bias[j];
        }
    }
    
    free(qkv);
    free(attn_output);
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

void feed_forward(float *input, float *output, float *c_fc_weight, float *c_fc_bias,
                 float *c_proj_weight, float *c_proj_bias, int seq_len, int d_model, int d_ff) {
    
    float *fc_output = malloc(seq_len * d_ff * sizeof(float));
    float *gelu_output = malloc(seq_len * d_ff * sizeof(float));
    
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

void transformer_block(float *input, float *output, Tensor *tensors, int num_tensors,
                      int layer_idx, int seq_len, int d_model, int num_heads, int d_k, int d_ff, float epsilon) {
    
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
    
    float *ln_1_output = malloc(seq_len * d_model * sizeof(float));
    float *attn_output = malloc(seq_len * d_model * sizeof(float));
    float *Q = malloc(seq_len * d_model * sizeof(float));
    float *K = malloc(seq_len * d_model * sizeof(float));
    float *V = malloc(seq_len * d_model * sizeof(float));
    float *residual_1 = malloc(seq_len * d_model * sizeof(float));
    float *ffn_output = malloc(seq_len * d_model * sizeof(float));
    float *ln_2_output = malloc(seq_len * d_model * sizeof(float));
    
    // First LayerNorm
    layer_norm(input, ln_1_output, ln_1_weight->data, ln_1_bias->data, seq_len, d_model, epsilon);
    
    // Self-Attention
    self_attention(ln_1_output, attn_output, c_attn_weight->data, c_attn_bias->data,
                  c_proj_weight->data, c_proj_bias->data, seq_len, d_model,
                  num_heads, d_k, Q, K, V);
    
    // Residual connection: input + attn_output
    for (int i = 0; i < seq_len * d_model; i++) {
        residual_1[i] = input[i] + attn_output[i];
    }
    
    // Second LayerNorm
    layer_norm(residual_1, ln_2_output, ln_2_weight->data, ln_2_bias->data, seq_len, d_model, epsilon);
    
    // Feed-Forward
    feed_forward(ln_2_output, ffn_output, c_fc_weight->data, c_fc_bias->data,
                c_proj_mlp_weight->data, c_proj_mlp_bias->data, seq_len, d_model, d_ff);
    
    // Residual connection and output: residual_1 + ffn_output
    for (int i = 0; i < seq_len * d_model; i++) {
        output[i] = residual_1[i] + ffn_output[i];
    }
    
    free(ln_1_output);
    free(attn_output);
    free(Q);
    free(K);
    free(V);
    free(residual_1);
    free(ffn_output);
    free(ln_2_output);
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
    return "<unknown>";
}

void printclean(int *initial_tokens, int initial_len, TokenEntry *token_mapping, int num_token_entries) {
	for (int i = 0; i < initial_len; i++) {
		const char *token_str = get_token_str(token_mapping, num_token_entries, initial_tokens[i]);
		// Process each character to replace Ġ/Ċ
		for (int j = 0; token_str[j] != '\0'; ) {
			if ((unsigned char)token_str[j] == 0xC4) {
				if ((unsigned char)token_str[j+1] == 0xA0) {  // Ġ -> space
					putchar(' ');
					j += 2;
				} else if ((unsigned char)token_str[j+1] == 0x8A) {  // Ċ -> newline
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
        fflush(stdout);  // Ensure immediate output
}


// New function to generate text
// Add these functions to your code for better text generation
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
    // In production code, use a more efficient algorithm
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

// Modify your generate_text function to use these sampling methods
void generate_text(Tensor *tensors, int num_tensors, int *initial_tokens, int initial_len, 
                  int max_new_tokens, TokenEntry *token_mapping, int num_token_entries) {
    
    // Initialize random seed
    srand(time(NULL));
    
    // Hyperparameters for sampling
    float temperature = 0.7f;  // Lower = more focused, higher = more random
    int top_k = 40;            // Consider only top k tokens
    float top_p = 0.9f;        // Consider tokens with cumulative probability < p
    
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
    Timer timer;
    start_timer(&timer);
    // Generate new tokens
    for (int iter = 0; iter < max_new_tokens; iter++) {
        // Compute embeddings
        float *embeddings = malloc(current_seq_len * d_model * sizeof(float));
        for (int i = 0; i < current_seq_len; i++) {
            for (int j = 0; j < d_model; j++) {
                embeddings[i * d_model + j] = wte->data[token_sequence[i] * d_model + j] +
                                             wpe->data[i * d_model + j];
            }
        }
        
        // Forward pass through all layers
        float *layer_input = malloc(current_seq_len * d_model * sizeof(float));
        float *layer_output = malloc(current_seq_len * d_model * sizeof(float));
        
        // Copy embeddings to layer_input
        memcpy(layer_input, embeddings, current_seq_len * d_model * sizeof(float));
        
        // Process each layer
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
            transformer_block(layer_input, layer_output, tensors, num_tensors,
                             layer_idx, current_seq_len, d_model, num_heads, d_k, d_ff, epsilon);
            // Swap input and output for next layer
            float *temp = layer_input;
            layer_input = layer_output;
            layer_output = temp;
        }
        
        // Compute logits for the last position only
        float *logits = malloc(vocab_size * sizeof(float));
        float *probs = malloc(vocab_size * sizeof(float));
        float *ln_output = malloc(d_model * sizeof(float));
        
        // Apply layer norm
        layer_norm(&layer_input[(current_seq_len-1) * d_model], ln_output, 
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
        free(embeddings);
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
    free(token_sequence);
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
    
    // Generate 50 new tokens
    generate_text(tensors, num_tensors, token_ids, seq_len, 10, token_mapping, num_token_entries);
    
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

