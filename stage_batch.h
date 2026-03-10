#ifndef STAGE_BATCH_H
#define STAGE_BATCH_H

#include "app.h"

int stage_worker_init(struct stage_worker *w);
void stage_worker_fini(struct stage_worker *w);

void stage_submit_batch(void *arg);

#endif