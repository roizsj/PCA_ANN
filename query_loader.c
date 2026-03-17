#include "query_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static inline float safe_sqrtf(float x) {
    if (x <= 0.0f) return 1.0f;
    return sqrtf(x);
}

int load_fvecs(const char *path, float **data_out, uint32_t *n_out, uint32_t *dim_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen fvecs");
        return -1;
    }

    int dim_i = 0;
    if (fread(&dim_i, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "[load_fvecs] failed to read first dim from %s\n", path);
        fclose(fp);
        return -1;
    }

    if (dim_i <= 0) {
        fprintf(stderr, "[load_fvecs] invalid dim=%d in %s\n", dim_i, path);
        fclose(fp);
        return -1;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        perror("fseek end");
        fclose(fp);
        return -1;
    }

    long fsize = ftell(fp);
    if (fsize < 0) {
        perror("ftell");
        fclose(fp);
        return -1;
    }

    rewind(fp);

    long rec_size = (long)sizeof(int) + (long)dim_i * (long)sizeof(float);
    if (rec_size <= 0 || fsize % rec_size != 0) {
        fprintf(stderr, "[load_fvecs] invalid file size=%ld rec_size=%ld for %s\n",
                fsize, rec_size, path);
        fclose(fp);
        return -1;
    }

    uint32_t n = (uint32_t)(fsize / rec_size);
    float *data = (float *)malloc((size_t)n * (size_t)dim_i * sizeof(float));
    if (!data) {
        fprintf(stderr, "[load_fvecs] malloc failed\n");
        fclose(fp);
        return -1;
    }

    for (uint32_t i = 0; i < n; i++) {
        int d = 0;
        if (fread(&d, sizeof(int), 1, fp) != 1) {
            fprintf(stderr, "[load_fvecs] failed to read dim for vec %u\n", i);
            free(data);
            fclose(fp);
            return -1;
        }

        if (d != dim_i) {
            fprintf(stderr, "[load_fvecs] inconsistent dim at vec %u: got %d expected %d\n",
                    i, d, dim_i);
            free(data);
            fclose(fp);
            return -1;
        }

        if (fread(&data[(size_t)i * (size_t)dim_i], sizeof(float), (size_t)dim_i, fp) != (size_t)dim_i) {
            fprintf(stderr, "[load_fvecs] failed to read vec %u\n", i);
            free(data);
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);

    *data_out = data;
    *n_out = n;
    *dim_out = (uint32_t)dim_i;

    printf("[load_fvecs] path=%s n=%u dim=%u\n", path, n, (uint32_t)dim_i);
    return 0;
}

int load_pca_mean(const char *path, float **mean_out, uint32_t *dim_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen pca_mean");
        return -1;
    }

    uint32_t dim = 0;
    if (fread(&dim, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "[load_pca_mean] failed to read dim\n");
        fclose(fp);
        return -1;
    }

    float *mean = (float *)malloc((size_t)dim * sizeof(float));
    if (!mean) {
        fprintf(stderr, "[load_pca_mean] malloc failed\n");
        fclose(fp);
        return -1;
    }

    if (fread(mean, sizeof(float), dim, fp) != dim) {
        fprintf(stderr, "[load_pca_mean] failed to read values\n");
        free(mean);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    *mean_out = mean;
    *dim_out = dim;

    printf("[load_pca_mean] dim=%u\n", dim);
    return 0;
}

int load_pca_components(const char *path, float **components_out, uint32_t *rows_out, uint32_t *cols_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen pca_components");
        return -1;
    }

    uint32_t rows = 0, cols = 0;
    if (fread(&rows, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&cols, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "[load_pca_components] failed to read header\n");
        fclose(fp);
        return -1;
    }

    float *components = (float *)malloc((size_t)rows * (size_t)cols * sizeof(float));
    if (!components) {
        fprintf(stderr, "[load_pca_components] malloc failed\n");
        fclose(fp);
        return -1;
    }

    size_t need = (size_t)rows * (size_t)cols;
    if (fread(components, sizeof(float), need, fp) != need) {
        fprintf(stderr, "[load_pca_components] failed to read body\n");
        free(components);
        fclose(fp);
        return -1;
    }

    fclose(fp);

    *components_out = components;
    *rows_out = rows;
    *cols_out = cols;

    printf("[load_pca_components] rows=%u cols=%u\n", rows, cols);
    return 0;
}

int load_pca_explained_variance(const char *path, float **var_out, uint32_t *dim_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen pca_explained_variance");
        return -1;
    }

    uint32_t dim = 0;
    if (fread(&dim, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "[load_pca_explained_variance] failed to read dim\n");
        fclose(fp);
        return -1;
    }

    float *var = (float *)malloc((size_t)dim * sizeof(float));
    if (!var) {
        fprintf(stderr, "[load_pca_explained_variance] malloc failed\n");
        fclose(fp);
        return -1;
    }

    if (fread(var, sizeof(float), dim, fp) != dim) {
        fprintf(stderr, "[load_pca_explained_variance] failed to read values\n");
        free(var);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    *var_out = var;
    *dim_out = dim;

    printf("[load_pca_explained_variance] dim=%u\n", dim);
    return 0;
}

int load_pca_meta(const char *path, uint32_t *n_components_out, uint32_t *original_dim_out, uint32_t *whiten_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen pca_meta");
        return -1;
    }

    uint32_t n_components = 0, original_dim = 0, whiten = 0;
    if (fread(&n_components, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&original_dim, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&whiten, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "[load_pca_meta] failed to read meta\n");
        fclose(fp);
        return -1;
    }

    fclose(fp);

    *n_components_out = n_components;
    *original_dim_out = original_dim;
    *whiten_out = whiten;

    printf("[load_pca_meta] n_components=%u original_dim=%u whiten=%u\n",
           n_components, original_dim, whiten);
    return 0;
}

int load_pca_model(const char *mean_path,
                   const char *components_path,
                   const char *explained_variance_path,
                   const char *meta_path,
                   pca_model_t *model) {
    memset(model, 0, sizeof(*model));

    uint32_t mean_dim = 0;
    uint32_t rows = 0, cols = 0;
    uint32_t var_dim = 0;
    uint32_t n_components = 0, original_dim = 0, whiten = 0;

    if (load_pca_meta(meta_path, &n_components, &original_dim, &whiten) != 0) {
        return -1;
    }

    if (load_pca_mean(mean_path, &model->mean, &mean_dim) != 0) {
        free_pca_model(model);
        return -1;
    }

    if (load_pca_components(components_path, &model->components, &rows, &cols) != 0) {
        free_pca_model(model);
        return -1;
    }

    if (load_pca_explained_variance(explained_variance_path, &model->explained_variance, &var_dim) != 0) {
        free_pca_model(model);
        return -1;
    }

    if (mean_dim != original_dim) {
        fprintf(stderr, "[load_pca_model] mean_dim=%u != original_dim=%u\n", mean_dim, original_dim);
        free_pca_model(model);
        return -1;
    }

    if (rows != n_components || cols != original_dim) {
        fprintf(stderr, "[load_pca_model] components shape mismatch: got [%u,%u], expected [%u,%u]\n",
                rows, cols, n_components, original_dim);
        free_pca_model(model);
        return -1;
    }

    if (var_dim != n_components) {
        fprintf(stderr, "[load_pca_model] explained_variance dim=%u != n_components=%u\n",
                var_dim, n_components);
        free_pca_model(model);
        return -1;
    }

    model->original_dim = original_dim;
    model->n_components = n_components;
    model->whiten = whiten;

    return 0;
}

void free_pca_model(pca_model_t *model) {
    if (!model) return;
    free(model->mean);
    free(model->components);
    free(model->explained_variance);
    memset(model, 0, sizeof(*model));
}

int transform_queries_with_pca(const float *queries,
                               uint32_t n_queries,
                               uint32_t query_dim,
                               const pca_model_t *model,
                               float **queries_out,
                               uint32_t *out_dim) {
    if (!queries || !model || !queries_out || !out_dim) {
        fprintf(stderr, "[transform_queries_with_pca] null input\n");
        return -1;
    }

    if (query_dim != model->original_dim) {
        fprintf(stderr, "[transform_queries_with_pca] query_dim=%u != model original_dim=%u\n",
                query_dim, model->original_dim);
        return -1;
    }

    uint32_t din = model->original_dim;
    uint32_t dout = model->n_components;

    float *out = (float *)malloc((size_t)n_queries * (size_t)dout * sizeof(float));
    if (!out) {
        fprintf(stderr, "[transform_queries_with_pca] malloc failed\n");
        return -1;
    }

    for (uint32_t i = 0; i < n_queries; i++) {
        const float *x = &queries[(size_t)i * (size_t)din];
        float *y = &out[(size_t)i * (size_t)dout];

        for (uint32_t k = 0; k < dout; k++) {
            const float *comp_k = &model->components[(size_t)k * (size_t)din];
            float sum = 0.0f;

            for (uint32_t j = 0; j < din; j++) {
                sum += (x[j] - model->mean[j]) * comp_k[j];
            }

            if (model->whiten) {
                float scale = safe_sqrtf(model->explained_variance[k]);
                sum /= scale;
            }

            y[k] = sum;
        }
    }

    *queries_out = out;
    *out_dim = dout;
    return 0;
}

int prepare_queries_with_pca(const char *fvecs_path,
                             const char *mean_path,
                             const char *components_path,
                             const char *explained_variance_path,
                             const char *meta_path,
                             query_set_t *qs) {
    if (!qs) return -1;
    memset(qs, 0, sizeof(*qs));

    float *raw_queries = NULL;
    uint32_t nq = 0, raw_dim = 0;
    pca_model_t model;

    memset(&model, 0, sizeof(model));

    if (load_fvecs(fvecs_path, &raw_queries, &nq, &raw_dim) != 0) {
        return -1;
    }

    if (load_pca_model(mean_path, components_path, explained_variance_path, meta_path, &model) != 0) {
        free(raw_queries);
        return -1;
    }

    if (transform_queries_with_pca(raw_queries, nq, raw_dim, &model, &qs->data, &qs->dim) != 0) {
        free(raw_queries);
        free_pca_model(&model);
        return -1;
    }

    qs->n_queries = nq;

    free(raw_queries);
    free_pca_model(&model);

    printf("[prepare_queries_with_pca] nq=%u transformed_dim=%u\n", qs->n_queries, qs->dim);
    return 0;
}

void free_query_set(query_set_t *qs) {
    if (!qs) return;
    free(qs->data);
    memset(qs, 0, sizeof(*qs));
}