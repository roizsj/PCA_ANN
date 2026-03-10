#include "distance.h"

float compute_partial_l2_from_slot(const void *slot_buf, const float *query_seg)
{
    const float *x = (const float *)slot_buf;
    float sum = 0.0f;

    for (int i = 0; i < SEG_DIM; i++) {
        float d = x[i] - query_seg[i];
        sum += d * d;
    }
    return sum;
}

float fake_value(uint32_t vec_id, uint8_t stage, int dim_in_seg)
{
    return (float)(vec_id * 0.1f + stage + dim_in_seg * 0.01f);
}

float reference_partial_distance(uint32_t vec_id, uint8_t stage, const float *query_seg)
{
    float sum = 0.0f;
    for (int i = 0; i < SEG_DIM; i++) {
        float x = fake_value(vec_id, stage, i);
        float d = x - query_seg[i];
        sum += d * d;
    }
    return sum;
}

float reference_full_distance(uint32_t vec_id, float *query_segs[NUM_STAGES])
{
    float sum = 0.0f;
    for (int s = 0; s < NUM_STAGES; s++) {
        sum += reference_partial_distance(vec_id, s, query_segs[s]);
    }
    return sum;
}