#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/cJSON.h"

// Create token_mapping.txt from vocab.json
int main(int argc, char **argv) {
    /* if (argc != 3) { */
    /*     fprintf(stderr, "Usage: %s vocab.json token_mapping.txt\n", argv[0]); */
    /*     return 1; */
    /* } */
    
    /* const char *vocab_file = argv[1]; */
    /* const char *output_file = argv[2]; */
    const char *vocab_file = "Model/vocab.json";
    const char *output_file = "token_mapping.txt";
    
    // Load vocab.json
    FILE *f = fopen(vocab_file, "rb");
    if (!f) { 
        fprintf(stderr, "Failed to open %s\n", vocab_file); 
        return 1; 
    }
    
    // Get file size
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // Read file content
    char *data = malloc(len + 1);
    if (!data) { 
        fprintf(stderr, "Memory allocation failed\n"); 
        fclose(f);
        return 1; 
    }
    fread(data, 1, len, f);
    data[len] = '\0';
    fclose(f);
    
    // Parse JSON
    cJSON *root = cJSON_Parse(data);
    free(data);
    
    if (!root) { 
        fprintf(stderr, "Failed to parse %s\n", vocab_file); 
        return 1; 
    }
    
    // Open output file
    FILE *out = fopen(output_file, "w");
    if (!out) {
        fprintf(stderr, "Failed to create %s\n", output_file);
        cJSON_Delete(root);
        return 1;
    }
    
    // Write header
    fprintf(out, "# Token ID to string mapping\n");
    fprintf(out, "# ID TOKEN\n");
    
    // Iterate through the JSON object
    cJSON *token;
    cJSON_ArrayForEach(token, root) {
        int id = token->valueint;
        const char *token_str = token->string;
        
        // Write to file: ID TOKEN
        fprintf(out, "%d %s\n", id, token_str);
    }
    
    printf("Created token mapping file: %s\n", output_file);
    
    fclose(out);
    cJSON_Delete(root);
    return 0;
}

