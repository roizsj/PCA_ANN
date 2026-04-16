import argparse
import os
from typing import Optional

import numpy as np


def read_fvecs(path: str, mmap: bool = True) -> np.ndarray:
    raw = np.memmap(path, dtype=np.int32, mode="r") if mmap else np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"empty fvecs file: {path}")

    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"invalid fvecs dim={dim}: {path}")

    row_size = dim + 1
    if raw.size % row_size != 0:
        raise ValueError(f"invalid fvecs size: total_int32={raw.size}, row_size={row_size}")

    rows = raw.reshape(-1, row_size)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"inconsistent row dims in {path}")

    vecs = rows[:, 1:].view(np.float32)
    return vecs if mmap else vecs.copy()


def save_centroids(path: str, centroids: np.ndarray) -> None:
    centroids = np.asarray(centroids, dtype=np.float32)
    if centroids.ndim != 2:
        raise ValueError("centroids must be 2D")
    with open(path, "wb") as f:
        np.array([centroids.shape[0], centroids.shape[1]], dtype=np.uint32).tofile(f)
        centroids.tofile(f)


def load_centroids(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.uint32, count=2)
        if header.size != 2:
            raise ValueError(f"failed to read centroid header: {path}")
        nlist, dim = int(header[0]), int(header[1])
        body = np.fromfile(f, dtype=np.float32)
    if body.size != nlist * dim:
        raise ValueError(f"centroid body mismatch: expected={nlist * dim}, got={body.size}")
    return body.reshape(nlist, dim)


def save_labels(path: str, labels: np.ndarray) -> None:
    labels = np.asarray(labels, dtype=np.uint32)
    if labels.ndim != 1:
        raise ValueError("labels must be 1D")
    with open(path, "wb") as f:
        np.array([labels.size], dtype=np.uint32).tofile(f)
        labels.tofile(f)


def save_members(path: str, offsets: np.ndarray, ids: np.ndarray) -> None:
    offsets = np.asarray(offsets, dtype=np.uint64)
    ids = np.asarray(ids, dtype=np.uint32)
    if offsets.ndim != 1 or ids.ndim != 1:
        raise ValueError("offsets and ids must be 1D")
    if offsets[-1] != ids.size:
        raise ValueError(f"offset/id mismatch: offsets[-1]={offsets[-1]}, ids={ids.size}")

    with open(path, "wb") as f:
        np.array([offsets.size - 1, ids.size], dtype=np.uint64).tofile(f)
        offsets.tofile(f)
        ids.tofile(f)


def maybe_sample(x: np.ndarray, train_size: int, seed: int) -> np.ndarray:
    n = x.shape[0]
    if train_size <= 0 or train_size >= n:
        return np.ascontiguousarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=train_size, replace=False)
    return np.ascontiguousarray(x[idx], dtype=np.float32)


def train_faiss(
    x: np.ndarray,
    nlist: int,
    train_size: int,
    niter: int,
    seed: int,
    use_gpu: bool,
) -> np.ndarray:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss is required for practical GIST-1M x 50k clustering") from exc

    train_x = maybe_sample(x, train_size, seed)
    dim = train_x.shape[1]
    print(f"[train] faiss kmeans n_train={train_x.shape[0]} dim={dim} nlist={nlist} niter={niter}")

    kmeans = faiss.Kmeans(
        dim,
        nlist,
        niter=niter,
        verbose=True,
        seed=seed,
        gpu=use_gpu,
    )
    kmeans.cp.min_points_per_centroid = 1
    kmeans.train(train_x)
    centroids = kmeans.centroids
    if not isinstance(centroids, np.ndarray):
        centroids = faiss.vector_to_array(centroids)
    return np.asarray(centroids, dtype=np.float32).reshape(nlist, dim)


def make_faiss_index(centroids: np.ndarray, use_gpu: bool):
    import faiss

    index = faiss.IndexFlatL2(centroids.shape[1])
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index._gpu_resources = res
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    return index


def balanced_assign(
    x: np.ndarray,
    centroids: np.ndarray,
    target_size: int,
    max_size: int,
    candidate_k: int,
    chunk_size: int,
    seed: int,
    use_gpu: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if max_size < target_size:
        raise ValueError(f"max_size={max_size} must be >= target_size={target_size}")

    n, dim = x.shape
    nlist, cdim = centroids.shape
    if dim != cdim:
        raise ValueError(f"dim mismatch: x_dim={dim}, centroid_dim={cdim}")
    if n > nlist * max_size:
        raise ValueError(f"capacity too small: n={n}, nlist*max_size={nlist * max_size}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    labels = np.full(n, np.iinfo(np.uint32).max, dtype=np.uint32)
    counts = np.zeros(nlist, dtype=np.int32)
    index = make_faiss_index(centroids, use_gpu)

    print(
        f"[assign] balanced greedy n={n} nlist={nlist} target={target_size} "
        f"max={max_size} candidate_k={candidate_k} chunk={chunk_size}"
    )

    for pos in range(0, n, chunk_size):
        batch_ids = order[pos : pos + chunk_size]
        xb = np.ascontiguousarray(x[batch_ids], dtype=np.float32)
        _, candidates = index.search(xb, candidate_k)

        for row, vec_id in enumerate(batch_ids):
            chosen: Optional[int] = None
            for cid in candidates[row]:
                cid = int(cid)
                if cid >= 0 and counts[cid] < max_size:
                    chosen = cid
                    break

            if chosen is None:
                underfull = np.flatnonzero(counts < max_size)
                if underfull.size == 0:
                    raise RuntimeError("no cluster capacity left during assignment")
                chosen = int(underfull[np.argmin(counts[underfull])])

            labels[vec_id] = chosen
            counts[chosen] += 1

        print(f"[assign] assigned {min(pos + chunk_size, n)}/{n}; min={counts.min()} max={counts.max()}")

    return labels, counts


def build_members_from_labels(labels: np.ndarray, nlist: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels.astype(np.int64), minlength=nlist).astype(np.uint64)
    offsets = np.empty(nlist + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    ids = np.empty(labels.size, dtype=np.uint32)
    cursor = offsets[:-1].copy()
    for vec_id, cid in enumerate(labels.astype(np.int64)):
        pos = int(cursor[cid])
        ids[pos] = vec_id
        cursor[cid] += 1
    return offsets, ids


def add_redundant_members(
    x: np.ndarray,
    centroids: np.ndarray,
    base_offsets: np.ndarray,
    base_ids: np.ndarray,
    target_size: int,
    candidate_k: int,
    chunk_size: int,
    seed: int,
    use_gpu: bool,
) -> tuple[np.ndarray, np.ndarray]:
    nlist = centroids.shape[0]
    members: list[list[int]] = []
    for cid in range(nlist):
        start, end = int(base_offsets[cid]), int(base_offsets[cid + 1])
        members.append(base_ids[start:end].astype(np.uint32).tolist())

    need = np.array([max(0, target_size - len(m)) for m in members], dtype=np.int32)
    if need.sum() == 0:
        return base_offsets, base_ids

    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    index = make_faiss_index(centroids, use_gpu)
    print(f"[overlap] padding underfull clusters with redundant ids; total_missing={int(need.sum())}")

    for pos in range(0, x.shape[0], chunk_size):
        if need.sum() == 0:
            break
        batch_ids = order[pos : pos + chunk_size]
        xb = np.ascontiguousarray(x[batch_ids], dtype=np.float32)
        _, candidates = index.search(xb, candidate_k)

        for row, vec_id in enumerate(batch_ids):
            for cid in candidates[row]:
                cid = int(cid)
                if cid >= 0 and need[cid] > 0:
                    members[cid].append(int(vec_id))
                    need[cid] -= 1
                    break
            if need.sum() == 0:
                break

        print(f"[overlap] remaining_missing={int(need.sum())}")

    if need.sum() != 0:
        raise RuntimeError(f"failed to pad all clusters; remaining_missing={int(need.sum())}")

    offsets = np.zeros(nlist + 1, dtype=np.uint64)
    for cid, cluster_ids in enumerate(members):
        offsets[cid + 1] = offsets[cid] + len(cluster_ids)
    ids = np.empty(int(offsets[-1]), dtype=np.uint32)
    for cid, cluster_ids in enumerate(members):
        ids[int(offsets[cid]) : int(offsets[cid + 1])] = np.asarray(cluster_ids, dtype=np.uint32)
    return offsets, ids


def print_stats(name: str, counts: np.ndarray) -> None:
    print(f"[stats] {name}")
    print(f"  min={int(counts.min())} max={int(counts.max())} mean={counts.mean():.2f}")
    print(f"  p01={np.percentile(counts, 1):.1f} p50={np.percentile(counts, 50):.1f} p99={np.percentile(counts, 99):.1f}")
    print(f"  empty={int((counts == 0).sum())}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a size-balanced GIST-1M IVF partition, defaulting to 50k clusters of about 20 vectors"
    )
    parser.add_argument("--input", default="/home/zhangshujie/ann_nic/gist/gist_base.fvecs")
    parser.add_argument("--outdir", default="/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_balanced_ivf_50k")
    parser.add_argument("--nlist", type=int, default=50000)
    parser.add_argument("--target-size", type=int, default=20)
    parser.add_argument("--max-size", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=64, help="nearest centroid candidates considered per vector")
    parser.add_argument("--assign-chunk", type=int, default=20000)
    parser.add_argument("--train-size", type=int, default=0, help="0 means train on all vectors")
    parser.add_argument("--train-iter", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--centroids-in", default=None, help="reuse existing centroids and only rebalance assignments")
    parser.add_argument("--gpu", action="store_true", help="use faiss GPU resources for training/search")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--allow-redundant-members", action="store_true", help="pad underfull membership lists by duplicating vector ids")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    x = read_fvecs(args.input, mmap=not args.no_mmap)
    n, dim = x.shape
    print(f"[load] input={args.input} n={n} dim={dim} dtype={x.dtype}")

    if args.centroids_in:
        centroids = load_centroids(args.centroids_in)
        if centroids.shape != (args.nlist, dim):
            raise ValueError(f"centroid shape={centroids.shape}, expected=({args.nlist},{dim})")
    else:
        centroids = train_faiss(x, args.nlist, args.train_size, args.train_iter, args.random_state, args.gpu)

    labels, counts = balanced_assign(
        x=x,
        centroids=centroids,
        target_size=args.target_size,
        max_size=args.max_size,
        candidate_k=args.candidate_k,
        chunk_size=args.assign_chunk,
        seed=args.random_state,
        use_gpu=args.gpu,
    )

    offsets, ids = build_members_from_labels(labels, args.nlist)
    if args.allow_redundant_members:
        offsets, ids = add_redundant_members(
            x=x,
            centroids=centroids,
            base_offsets=offsets,
            base_ids=ids,
            target_size=args.target_size,
            candidate_k=args.candidate_k,
            chunk_size=args.assign_chunk,
            seed=args.random_state + 1,
            use_gpu=args.gpu,
        )

    centroid_path = os.path.join(args.outdir, "gist_balanced_ivf_centroids.bin")
    labels_path = os.path.join(args.outdir, "gist_balanced_codebook.bin")
    members_path = os.path.join(args.outdir, "gist_balanced_members.bin")

    save_centroids(centroid_path, centroids)
    save_labels(labels_path, labels)
    save_members(members_path, offsets, ids)

    print_stats("single-assignment labels", counts)
    member_counts = np.diff(offsets)
    print_stats("membership lists", member_counts)
    print("[done] wrote:")
    print(f"  centroids: {centroid_path}")
    print(f"  labels:    {labels_path}")
    print(f"  members:   {members_path}")


if __name__ == "__main__":
    main()
