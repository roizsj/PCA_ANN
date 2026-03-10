#include "layout.h"

uint64_t
layout_lba_for_vec(uint32_t vec_id, uint8_t stage)
{
    (void)stage;
    /* 最简单：每盘都按 vec_id 顺序平铺 */
    return (uint64_t)vec_id * (IO_BYTES / 512);
}

uint32_t
layout_lba_count(struct stage_worker *w)
{
    uint32_t sector = spdk_nvme_ns_get_sector_size(w->ns);
    return IO_BYTES / sector;
}