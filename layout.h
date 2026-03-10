#ifndef LAYOUT_H
#define LAYOUT_H

#include "app.h"

uint64_t layout_lba_for_vec(uint32_t vec_id, uint8_t stage);
uint32_t layout_lba_count(struct stage_worker *w);

#endif