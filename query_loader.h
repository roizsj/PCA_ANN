#ifndef QUERY_LOADER_H
#define QUERY_LOADER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t original_dim;          // 128
    uint32_t n_components;          // 128
    uint32_t whiten;                // 0 or 1

    float *mean;                    // [original_dim]
    float *components;              // [n_components * original_dim]
    float *explained_variance;      // [n_components]
} pca_model_t;

typedef struct {
    uint32_t n_queries;
    uint32_t dim;                   // transformed dim = n_components
    float *data;                    // [n_queries * dim]
} query_set_t;

/* fvecs */
int load_fvecs(const char *path, float **data_out, uint32_t *n_out, uint32_t *dim_out);

/* PCA model loaders */
int load_pca_mean(const char *path, float **mean_out, uint32_t *dim_out);
int load_pca_components(const char *path, float **components_out, uint32_t *rows_out, uint32_t *cols_out);
int load_pca_explained_variance(const char *path, float **var_out, uint32_t *dim_out);
int load_pca_meta(const char *path, uint32_t *n_components_out, uint32_t *original_dim_out, uint32_t *whiten_out);

int load_pca_model(const char *mean_path,
                   const char *components_path,
                   const char *explained_variance_path,
                   const char *meta_path,
                   pca_model_t *model);

void free_pca_model(pca_model_t *model);

/* transform */
int transform_queries_with_pca(const float *queries,
                               uint32_t n_queries,
                               uint32_t query_dim,
                               const pca_model_t *model,
                               float **queries_out,
                               uint32_t *out_dim);

/* one-stop */
int prepare_queries_with_pca(const char *fvecs_path,
                             const char *mean_path,
                             const char *components_path,
                             const char *explained_variance_path,
                             const char *meta_path,
                             query_set_t *qs);

void free_query_set(query_set_t *qs);

#ifdef __cplusplus
}
#endif

#endif