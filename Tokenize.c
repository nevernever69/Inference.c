#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "cJSON.h"
#include "Tokenize.h"

#define MAX_TOKEN_LEN 128

typedef struct {
    char *token;
    int id;
    UT_hash_handle hh;
} VocabEntry;

typedef struct {
    char *pair;
    int rank;
    UT_hash_handle hh;
} MergeEntry;

static VocabEntry *vocab = NULL;
static MergeEntry *merges = NULL;

void free_vocab(void) {
    VocabEntry *entry, *tmp;
    HASH_ITER(hh, vocab, entry, tmp) {
        HASH_DEL(vocab, entry);
        free(entry->token);
        free(entry);
    }
}

void free_merges(void) {
    MergeEntry *entry, *tmp;
    HASH_ITER(hh, merges, entry, tmp) {
        HASH_DEL(merges, entry);
        free(entry->pair);
        free(entry);
    }
}

void load_vocab(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *data = malloc(len + 1);
    if (!data) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    fread(data, 1, len, f);
    data[len] = '\0';
    fclose(f);

    cJSON *root = cJSON_Parse(data);
    free(data);
    if (!root) { fprintf(stderr, "Failed to parse %s\n", path); exit(1); }
    cJSON *item = NULL;
    cJSON_ArrayForEach(item, root) {
        VocabEntry *e = malloc(sizeof(VocabEntry));
        if (!e) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
        e->token = strdup(item->string);
        e->id = item->valueint;
        HASH_ADD_KEYPTR(hh, vocab, e->token, strlen(e->token), e);
    }
    cJSON_Delete(root);
}

void load_merges(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); exit(1); }
    char buf[256];
    int rank = 0;
    fgets(buf, sizeof(buf), f);
    while (fgets(buf, sizeof(buf), f)) {
        char *a = strtok(buf, " \n");
        char *b = strtok(NULL, " \n");
        if (!a || !b) break;
        char pair[MAX_TOKEN_LEN];
        snprintf(pair, sizeof(pair), "%s %s", a, b);
        MergeEntry *e = malloc(sizeof(MergeEntry));
        e->pair = strdup(pair);
        e->rank = rank++;
        HASH_ADD_KEYPTR(hh, merges, e->pair, strlen(e->pair), e);
    }
    fclose(f);
}

char **bpe_encode(const char *word, int *out_len) {
    int n = strlen(word);
    char **symbols = malloc((n + 1) * sizeof(char *));
    if (!symbols) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    for (int i = 0; i < n; i++) {
        symbols[i] = malloc(3);
        if (!symbols[i]) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
        unsigned char c = word[i];
        sprintf(symbols[i], "%c", c);
    }
    *out_len = n;

    while (1) {
        int best_rank = INT_MAX;
        int best_i = -1;
        for (int i = 0; i < *out_len - 1; i++) {
            char pair[MAX_TOKEN_LEN];
            snprintf(pair, sizeof(pair), "%s %s", symbols[i], symbols[i + 1]);
            MergeEntry *m;
            HASH_FIND_STR(merges, pair, m);
            if (m && m->rank < best_rank) {
                best_rank = m->rank;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        char *a = symbols[best_i];
        char *b = symbols[best_i + 1];
        char merged[MAX_TOKEN_LEN];
        snprintf(merged, sizeof(merged), "%s%s", a, b);
        free(symbols[best_i]);
        free(symbols[best_i + 1]);
        symbols[best_i] = strdup(merged);
        for (int i = best_i + 1; i < *out_len - 1; i++) {
            symbols[i] = symbols[i + 1];
        }
        (*out_len)--;
    }

    /* printf("BPE for %s: ", word); */
    /* for (int i = 0; i < *out_len; i++) { */
    /*     printf("%s ", symbols[i]); */
    /* } */
    /* printf("\n"); */
    return symbols;
}

int *tokenize(const char *input, int *out_len) {
    load_vocab("vocab.json");
    load_merges("merges.txt");
    char *s = strdup(input);
    if (!s) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    char *word = strtok(s, " \t\n");
    int count = 0;
    int *out = NULL;
    int out_size = 0;

    while (word) {
        /* printf("Processing word: %s\n", word); */
        int k;
        char **tokens = bpe_encode(word, &k);
        for (int i = 0; i < k; i++) {
            char *token = tokens[i];
            char normalized[MAX_TOKEN_LEN];
            // Add Ġ to the first subword of words after the first word
            if (i == 0 && count > 0) {
                snprintf(normalized, MAX_TOKEN_LEN, "Ġ%s", token);
            } else {
                snprintf(normalized, MAX_TOKEN_LEN, "%s", token);
            }

            VocabEntry *v;
            HASH_FIND_STR(vocab, normalized, v);
            if (!v) {
                HASH_FIND_STR(vocab, token, v); // Fallback to raw token
            }
            if (v) {
                if (count >= out_size) {
                    out_size = out_size == 0 ? 1 : out_size * 2;
                    out = realloc(out, out_size * sizeof(int));
                    if (!out) { fprintf(stderr, "Memory allocation failed\n"); free(s); free_vocab(); free_merges(); exit(1); }
                }
                out[count++] = v->id;
                /* printf("Token: %s (normalized: %s), ID: %d\n", token, v->token, v->id); */
            } else {
                fprintf(stderr, "OOV token %s (normalized: %s), using <|endoftext|> (ID 50256)\n", token, normalized);
                if (count >= out_size) {
                    out_size = out_size == 0 ? 1 : out_size * 2;
                    out = realloc(out, out_size * sizeof(int));
                    if (!out) { fprintf(stderr, "Memory allocation failed\n"); free(s); free_vocab(); free_merges(); exit(1); }
                }
                out[count++] = 50256;
            }
            free(tokens[i]);
        }
        free(tokens);
        word = strtok(NULL, " \t\n");
    }

    *out_len = count;
    free(s);
    free_vocab();
    free_merges();
    return out;
}


