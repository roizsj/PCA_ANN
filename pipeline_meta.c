#include "pipeline_stage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int parse_ivf_meta(const char *filename, ivf_meta_t *meta)
{
    if (!meta) {
        return -1;
    }
    memset(meta, 0, sizeof(*meta));

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen");
        return -1;
    }

    ivf_meta_header_t header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fprintf(stderr, "Failed to read MetaHeader\n");
        fclose(fp);
        return -1;
    }

    if (header.magic != IVF_META_MAGIC_FLEX) {
        fprintf(stderr, "Invalid magic number: 0x%08X\n", header.magic);
        fclose(fp);
        return -1;
    }

    if (header.version != 1) {
        fprintf(stderr, "Unsupported IVF meta version: %u\n", header.version);
        fclose(fp);
        return -1;
    }

    if (header.num_shards != NUM_STAGES) {
        fprintf(stderr, "Unsupported num_shards: %u expected=%d\n",
                header.num_shards, NUM_STAGES);
        fclose(fp);
        return -1;
    }

    if (header.dim == 0 || header.nlist == 0 || header.vectors_per_lba == 0) {
        fprintf(stderr,
                "Invalid flex IVF header: dim=%u nlist=%u vectors_per_lba=%u\n",
                header.dim, header.nlist, header.vectors_per_lba);
        fclose(fp);
        return -1;
    }

    meta->header = header;
    meta->nlist = header.nlist;

    meta->clusters = malloc(header.nlist * sizeof(cluster_info_t));
    if (!meta->clusters) {
        perror("malloc clusters");
        fclose(fp);
        return -1;
    }

    meta->centroids = malloc((size_t)header.nlist * (size_t)header.dim * sizeof(float));
    if (!meta->centroids) {
        perror("malloc centroids");
        free(meta->clusters);
        meta->clusters = NULL;
        fclose(fp);
        return -1;
    }

    uint32_t sorted_id_base = 0;
    for (uint32_t i = 0; i < header.nlist; i++) {
        ClusterMetaOnDisk cm;

        if (fread(&cm, sizeof(cm), 1, fp) != 1) {
            fprintf(stderr, "Failed to read cluster %u metadata\n", i);
            free_ivf_meta(meta);
            fclose(fp);
            return -1;
        }

        if (cm.cluster_id != i) {
            fprintf(stderr,
                    "parse_ivf_meta: unsupported cluster_id layout at idx=%u cluster_id=%u\n",
                    i, cm.cluster_id);
            free_ivf_meta(meta);
            fclose(fp);
            return -1;
        }

        meta->clusters[i].cluster_id = cm.cluster_id;
        meta->clusters[i].start_lba = cm.start_lba;
        meta->clusters[i].num_vectors = cm.num_vectors;
        meta->clusters[i].num_lbas = cm.num_lbas;
        meta->clusters[i].sorted_id_base = sorted_id_base;
        sorted_id_base += cm.num_vectors;
    }

    {
        size_t want = (size_t)header.nlist * (size_t)header.dim;
        if (fread(meta->centroids, sizeof(float), want, fp) != want) {
            fprintf(stderr, "Failed to read centroids payload from %s\n", filename);
            free_ivf_meta(meta);
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

int load_sorted_ids_bin(const char *path, uint32_t *num_vectors_out, uint32_t **sorted_ids_out)
{
    if (!path || !num_vectors_out || !sorted_ids_out) {
        return -1;
    }

    *num_vectors_out = 0;
    *sorted_ids_out = NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen sorted ids");
        return -1;
    }

    uint32_t num_vectors = 0;
    if (fread(&num_vectors, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "load_sorted_ids_bin: failed to read count\n");
        fclose(fp);
        return -1;
    }

    uint32_t *sorted_ids = (uint32_t *)malloc((size_t)num_vectors * sizeof(uint32_t));
    if (!sorted_ids) {
        perror("malloc sorted ids");
        fclose(fp);
        return -1;
    }

    if (fread(sorted_ids, sizeof(uint32_t), num_vectors, fp) != num_vectors) {
        fprintf(stderr, "load_sorted_ids_bin: failed to read body\n");
        free(sorted_ids);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    *num_vectors_out = num_vectors;
    *sorted_ids_out = sorted_ids;
    return 0;
}

void free_ivf_meta(ivf_meta_t *meta)
{
    if (!meta) {
        return;
    }

    free(meta->clusters);
    free(meta->centroids);
    meta->clusters = NULL;
    meta->centroids = NULL;
    meta->nlist = 0;
    memset(&meta->header, 0, sizeof(meta->header));
}
