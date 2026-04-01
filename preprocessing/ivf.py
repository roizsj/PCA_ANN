import argparse
import numpy as np
from sklearn.cluster import MiniBatchKMeans


DIM = 960


def read_fvecs(path: str) -> np.ndarray:
    """
    读取 .fvecs 文件
    格式：每条记录 [int32 dim][float32 * dim]
    返回 shape=(n, d), dtype=float32
    """
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"Empty file: {path}")

    dim = raw[0]
    if dim <= 0:
        raise ValueError(f"Invalid dim={dim} in {path}")
    if dim != DIM:
        raise ValueError(f"Expected dim={DIM}, got dim={dim}")

    row_size = dim + 1
    if raw.size % row_size != 0:
        raise ValueError(
            f"Invalid fvecs size: total_int32={raw.size}, row_size={row_size}"
        )

    raw = raw.reshape(-1, row_size)
    dims = raw[:, 0]
    if not np.all(dims == dim):
        raise ValueError("Inconsistent dimensions in fvecs file")

    vecs = raw[:, 1:].copy().view(np.float32)
    return vecs


def save_centroids_bin(path: str, centroids: np.ndarray) -> None:
    """
    保存 centroids.bin:
    [u32 nlist][u32 dim][float centroids[nlist * dim]]
    """
    centroids = np.asarray(centroids, dtype=np.float32)
    if centroids.ndim != 2:
        raise ValueError("centroids must be 2D")

    nlist, dim = centroids.shape
    header = np.array([nlist, dim], dtype=np.uint32)

    with open(path, "wb") as f:
        header.tofile(f)
        centroids.astype(np.float32).tofile(f)


def load_centroids_bin(path: str) -> np.ndarray:
    """
    读取 centroids.bin
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.uint32, count=2)
        if header.size != 2:
            raise ValueError("Failed to read centroid header")

        nlist, dim = int(header[0]), int(header[1])
        if dim != DIM:
            raise ValueError(f"Expected dim={DIM}, got {dim}")

        centroids = np.fromfile(f, dtype=np.float32)
        if centroids.size != nlist * dim:
            raise ValueError(
                f"Centroid size mismatch: expected {nlist * dim}, got {centroids.size}"
            )

    return centroids.reshape(nlist, dim)


def save_codebook_bin(path: str, labels: np.ndarray) -> None:
    """
    保存 codebook.bin:
    [u32 num_vectors][u32 label0][u32 label1]...[u32 labelN-1]
    """
    labels = np.asarray(labels, dtype=np.uint32)
    if labels.ndim != 1:
        raise ValueError("labels must be 1D")

    n = np.array([labels.shape[0]], dtype=np.uint32)
    with open(path, "wb") as f:
        n.tofile(f)
        labels.tofile(f)


def load_codebook_bin(path: str) -> np.ndarray:
    """
    读取 codebook.bin
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.uint32, count=1)
        if header.size != 1:
            raise ValueError("Failed to read codebook header")
        n = int(header[0])

        labels = np.fromfile(f, dtype=np.uint32)
        if labels.size != n:
            raise ValueError(f"Codebook size mismatch: expected {n}, got {labels.size}")

    return labels


def train_ivf(
    x: np.ndarray,
    nlist: int,
    batch_size: int = 10000,
    max_iter: int = 100,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    用 MiniBatchKMeans 训练 IVF centroids，并返回:
    centroids: shape=(nlist, dim)
    labels: shape=(n,)
    """
    if x.ndim != 2:
        raise ValueError("Input x must be 2D")

    n, d = x.shape
    if d != DIM:
        raise ValueError(f"Expected dim={DIM}, got {d}")

    print(f"[INFO] Training IVF on {n} vectors, dim={d}, nlist={nlist}")

    kmeans = MiniBatchKMeans(
        n_clusters=nlist,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state,
        n_init=10,
        compute_labels=True,
        verbose=1,
    )
    kmeans.fit(x)

    centroids = kmeans.cluster_centers_.astype(np.float32)
    labels = kmeans.labels_.astype(np.uint32)

    return centroids, labels


def assign_with_existing_centroids(
    x: np.ndarray,
    centroids: np.ndarray,
    chunk_size: int = 100000,
) -> np.ndarray:
    """
    如果已经有 centroids，只做 assignment。
    为了避免一次性构建超大距离矩阵，这里按 chunk 做。
    """
    x = np.asarray(x, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)

    n, d = x.shape
    nlist, d2 = centroids.shape
    if d != d2:
        raise ValueError(f"Dim mismatch: x={d}, centroids={d2}")

    labels = np.empty(n, dtype=np.uint32)

    print(f"[INFO] Assigning {n} vectors to {nlist} centroids")

    # 用平方距离:
    # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    centroid_norms = np.sum(centroids * centroids, axis=1)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xb = x[start:end]

        xb_norms = np.sum(xb * xb, axis=1, keepdims=True)
        dots = xb @ centroids.T
        dists = xb_norms + centroid_norms[None, :] - 2.0 * dots

        labels[start:end] = np.argmin(dists, axis=1).astype(np.uint32)

        print(f"[INFO] Assigned {end}/{n}")

    return labels


def print_cluster_stats(labels: np.ndarray, nlist: int) -> None:
    counts = np.bincount(labels, minlength=nlist)
    print("[INFO] Cluster size stats:")
    print(f"  min   = {counts.min()}")
    print(f"  max   = {counts.max()}")
    print(f"  mean  = {counts.mean():.2f}")
    print(f"  empty = {(counts == 0).sum()}")


def main():
    parser = argparse.ArgumentParser(
        description="Build IVF centroids.bin and codebook.bin from base.fvecs"
    )
    parser.add_argument("--input", required=True, help="Input base.fvecs")
    parser.add_argument("--nlist", type=int, required=True, help="Number of IVF clusters")
    parser.add_argument("--centroids-out", required=True, help="Output centroids.bin")
    parser.add_argument("--codebook-out", required=True, help="Output codebook.bin")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)

    # 可选：如果你以后只想基于已有 centroids 重新 assign
    parser.add_argument(
        "--centroids-in",
        default=None,
        help="Optional existing centroids.bin; if provided, skip training and only assign"
    )

    args = parser.parse_args()

    x = read_fvecs(args.input)
    print(f"[INFO] Loaded vectors: shape={x.shape}, dtype={x.dtype}")

    if args.centroids_in is not None:
        centroids = load_centroids_bin(args.centroids_in)
        labels = assign_with_existing_centroids(x, centroids)
        save_centroids_bin(args.centroids_out, centroids)
    else:
        centroids, labels = train_ivf(
            x=x,
            nlist=args.nlist,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
        save_centroids_bin(args.centroids_out, centroids)

    save_codebook_bin(args.codebook_out, labels)
    print_cluster_stats(labels, centroids.shape[0])

    print(f"[DONE] Saved centroids to: {args.centroids_out}")
    print(f"[DONE] Saved codebook  to: {args.codebook_out}")


if __name__ == "__main__":
    main()