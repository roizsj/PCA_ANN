#ifndef QUERY_CTX_H
#define QUERY_CTX_H

#include "app.h"

struct query_ctx *query_ctx_create(const struct query_input *in);
struct query_ctx *query_ctx_get(uint64_t qid);
void query_ctx_destroy(uint64_t qid);

#endif