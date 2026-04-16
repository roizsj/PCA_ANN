import argparse
import os
import struct

import numpy as np
from sklearn.cluster import MiniBatchKMeans


MAGIC_PQ = 0x47515049  # "IPQG" little-endian marker for GIST IVF-PQ


def read_fvecs(path: str, mmap: bool = False) -> np.ndarray:
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


def save_ivf_centroids(path: str, centroids: np.ndarray) -> None:
    centroids = np.asarray(centroids, dtype=np.float32)
    with open(path, "wb") as f:
        np.array([centroids.shape[0], centroids.shape[1]], dtype=np.uint32).tofile(f)
        centroids.tofile(f)


def load_ivf_centroids(path: str) -> np.ndarray:
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
    with open(path, "wb") as f:
        np.array([labels.size], dtype=np.uint32).tofile(f)
        labels.tofile(f)


def train_ivf(x: np.ndarray, nlist: int, batch_size: int, max_iter: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    print(f"[ivf] training n={x.shape[0]} dim={x.shape[1]} nlist={nlist}")
    km = MiniBatchKMeans(
        n_clusters=nlist,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=10,
        random_state=random_state,
        compute_labels=True,
        verbose=1,
    )
    km.fit(x)
    return km.cluster_centers_.astype(np.float32), km.labels_.astype(np.uint32)


def assign_ivf(x: np.ndarray, centroids: np.ndarray, chunk_size: int) -> np.ndarray:
    n = x.shape[0]
    labels = np.empty(n, dtype=np.uint32)
    centroid_norms = np.sum(centroids * centroids, axis=1)
    print(f"[ivf] assigning n={n} nlist={centroids.shape[0]} chunk={chunk_size}")

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xb = np.asarray(x[start:end], dtype=np.float32)
        dists = np.sum(xb * xb, axis=1, keepdims=True) + centroid_norms[None, :] - 2.0 * (xb @ centroids.T)
        labels[start:end] = np.argmin(dists, axis=1).astype(np.uint32)
        print(f"[ivf] assigned {end}/{n}")
    return labels


def train_pq(x: np.ndarray, m: int, ksub: int, train_size: int, batch_size: int, max_iter: int, random_state: int) -> np.ndarray:
    n, dim = x.shape
    if dim % m != 0:
        raise ValueError(f"dim={dim} must be divisible by pq_m={m}")
    subdim = dim // m

    if train_size > 0 and train_size < n:
        rng = np.random.default_rng(random_state)
        train_idx = rng.choice(n, size=train_size, replace=False)
        train_x = np.asarray(x[train_idx], dtype=np.float32)
    else:
        train_x = np.asarray(x, dtype=np.float32)

    codebooks = np.empty((m, ksub, subdim), dtype=np.float32)
    print(f"[pq] training n_train={train_x.shape[0]} dim={dim} m={m} subdim={subdim} ksub={ksub}")

    for part in range(m):
        s0 = part * subdim
        s1 = s0 + subdim
        print(f"[pq] train subquantizer {part + 1}/{m}, dims=[{s0},{s1})")
        km = MiniBatchKMeans(
            n_clusters=ksub,
            batch_size=batch_size,
            max_iter=max_iter,
            n_init=3,
            random_state=random_state + part,
            compute_labels=False,
            verbose=0,
        )
        km.fit(train_x[:, s0:s1])
        codebooks[part] = km.cluster_centers_.astype(np.float32)

    return codebooks


def encode_pq(x: np.ndarray, codebooks: np.ndarray, chunk_size: int) -> np.ndarray:
    n, dim = x.shape
    m, ksub, subdim = codebooks.shape
    if dim != m * subdim:
        raise ValueError(f"dim mismatch: x_dim={dim}, pq_dim={m * subdim}")
    if ksub > 256:
        raise ValueError("this script currently writes uint8 PQ codes, so ksub must be <= 256")

    codes = np.empty((n, m), dtype=np.uint8)
    print(f"[pq] encoding n={n} m={m} chunk={chunk_size}")

    cb_norms = np.sum(codebooks * codebooks, axis=2)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xb = np.asarray(x[start:end], dtype=np.float32)
        for part in range(m):
            s0 = part * subdim
            s1 = s0 + subdim
            sub = xb[:, s0:s1]
            dists = np.sum(sub * sub, axis=1, keepdims=True) + cb_norms[part][None, :] - 2.0 * (sub @ codebooks[part].T)
            codes[start:end, part] = np.argmin(dists, axis=1).astype(np.uint8)
        print(f"[pq] encoded {end}/{n}")

    return codes


def save_pq_table(path: str, codebooks: np.ndarray, nbits: int) -> None:
    codebooks = np.asarray(codebooks, dtype=np.float32)
    m, ksub, subdim = codebooks.shape
    dim = m * subdim
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIIII", MAGIC_PQ, dim, m, ksub, nbits, subdim))
        codebooks.tofile(f)


def save_pq_codes(path: str, codes: np.ndarray) -> None:
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2:
        raise ValueError("codes must be 2D")
    with open(path, "wb") as f:
        np.array([codes.shape[0], codes.shape[1]], dtype=np.uint32).tofile(f)
        codes.tofile(f)


def print_cluster_stats(labels: np.ndarray, nlist: int) -> None:
    counts = np.bincount(labels.astype(np.int64), minlength=nlist)
    print("[ivf] cluster stats")
    print(f"  min={counts.min()} max={counts.max()} mean={counts.mean():.2f} empty={(counts == 0).sum()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GIST-1M IVF-PQ without PCA and export C/SPDK baseline inputs")
    parser.add_argument("--input", default="/home/zhangshujie/ann_nic/gist/gist_base.fvecs", help="raw GIST base .fvecs")
    parser.add_argument("--outdir", default="/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--pq-m", type=int, default=60, help="number of PQ subquantizers; 960/60=16 dims each")
    parser.add_argument("--pq-nbits", type=int, default=8, choices=[8], help="currently only uint8 PQ codes are exported")
    parser.add_argument("--ivf-centroids-in", default=None, help="optional existing IVF centroids; skip IVF training and only assign")
    parser.add_argument("--pq-train-size", type=int, default=200000)
    parser.add_argument("--ivf-batch-size", type=int, default=10000)
    parser.add_argument("--pq-batch-size", type=int, default=10000)
    parser.add_argument("--ivf-max-iter", type=int, default=100)
    parser.add_argument("--pq-max-iter", type=int, default=100)
    parser.add_argument("--assign-chunk", type=int, default=50000)
    parser.add_argument("--encode-chunk", type=int, default=50000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mmap", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    x = read_fvecs(args.input, mmap=args.mmap)
    n, dim = x.shape
    ksub = 1 << args.pq_nbits
    print(f"[load] input={args.input} n={n} dim={dim} dtype={x.dtype}")

    if args.ivf_centroids_in:
        centroids = load_ivf_centroids(args.ivf_centroids_in)
        labels = assign_ivf(x, centroids, args.assign_chunk)
    else:
        centroids, labels = train_ivf(x, args.nlist, args.ivf_batch_size, args.ivf_max_iter, args.random_state)

    if centroids.shape != (args.nlist, dim):
        raise ValueError(f"unexpected centroid shape={centroids.shape}, expected=({args.nlist},{dim})")

    codebooks = train_pq(x, args.pq_m, ksub, args.pq_train_size, args.pq_batch_size, args.pq_max_iter, args.random_state)
    codes = encode_pq(x, codebooks, args.encode_chunk)

    centroid_path = os.path.join(args.outdir, "gist_ivfpq_centroids.bin")
    labels_path = os.path.join(args.outdir, "gist_ivfpq_codebook.bin")
    pq_table_path = os.path.join(args.outdir, "gist_pq_table.bin")
    pq_codes_path = os.path.join(args.outdir, "gist_pq_codes.bin")

    save_ivf_centroids(centroid_path, centroids)
    save_labels(labels_path, labels)
    save_pq_table(pq_table_path, codebooks, args.pq_nbits)
    save_pq_codes(pq_codes_path, codes)
    print_cluster_stats(labels, args.nlist)

    print("[done] wrote:")
    print(f"  IVF centroids: {centroid_path}")
    print(f"  IVF labels:    {labels_path}")
    print(f"  PQ table:      {pq_table_path}")
    print(f"  PQ codes:      {pq_codes_path}")


if __name__ == "__main__":
    main()
