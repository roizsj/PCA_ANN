#include "distance.h"

float
compute_partial_l2_from_slot(const void *slot_buf, const float *query_seg)
{
    const float *x = (const float *)slot_buf;
    float sum = 0.0f;

    for (int i = 0; i < SEG_DIM; i++) {
        float d = x[i] - query_seg[i];
        sum += d * d;
    }
    return sum;
}