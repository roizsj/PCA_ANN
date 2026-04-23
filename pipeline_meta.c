#include "pipeline_stage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;
    uint32_t num_shards;
    uint32_t shard_dims[NUM_STAGES];
    uint32_t shard_offsets[NUM_STAGES];
    uint32_t shard_bytes[NUM_STAGES];
    uint32_t vectors_per_lba;
    uint32_t nlist;
    uint32_t num_vectors;
    uint32_t sector_size;
    uint64_t base_lba;
} ivf_meta_header_v1_t;

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

    ivf_meta_header_v1_t disk_header;
    if (fread(&disk_header, sizeof(disk_header), 1, fp) != 1) {
        fprintf(stderr, "Failed to read MetaHeader\n");
        fclose(fp);
        return -1;
    }

    ivf_meta_header_t header;
    memset(&header, 0, sizeof(header));
    header.magic = disk_header.magic;
    header.version = disk_header.version;
    header.dim = disk_header.dim;
    header.num_shards = disk_header.num_shards;
    memcpy(header.shard_dims, disk_header.shard_dims, sizeof(header.shard_dims));
    memcpy(header.shard_offsets, disk_header.shard_offsets, sizeof(header.shard_offsets));
    memcpy(header.shard_bytes, disk_header.shard_bytes, sizeof(header.shard_bytes));
    header.vectors_per_lba = disk_header.vectors_per_lba;
    header.nlist = disk_header.nlist;
    header.num_vectors = disk_header.num_vectors;
    header.sector_size = disk_header.sector_size;
    header.base_lba = disk_header.base_lba;

    if (header.magic != IVF_META_MAGIC_FLEX) {
        fprintf(stderr, "Invalid magic number: 0x%08X\n", header.magic);
        fclose(fp);
        return -1;
    }

    if (header.version != 1 && header.version != 2) {
        fprintf(stderr, "Unsupported IVF meta version: %u\n", header.version);
        fclose(fp);
        return -1;
    }

    if (header.version >= 2) {
        if (fread(header.shard_vectors_per_lba,
                  sizeof(header.shard_vectors_per_lba[0]),
                  NUM_STAGES,
                  fp) != NUM_STAGES) {
            fprintf(stderr, "Failed to read per-shard vectors_per_lba\n");
            fclose(fp);
            return -1;
        }
    } else {
        for (uint32_t s = 0; s < NUM_STAGES; s++) {
            header.shard_vectors_per_lba[s] = header.vectors_per_lba;
        }
    }

    if (header.num_shards == 0 || header.num_shards > NUM_STAGES) {
        fprintf(stderr, "Unsupported num_shards: %u valid_range=[1,%d]\n",
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
    for (uint32_t s = 0; s < header.num_shards; s++) {
        if (header.shard_vectors_per_lba[s] == 0) {
            fprintf(stderr, "Invalid shard_vectors_per_lba[%u]=0\n", s);
            fclose(fp);
            return -1;
        }
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
