import argparse
import os
import numpy as np
from sklearn.decomposition import PCA


def read_fvecs(filename, mmap=False):
    """
    读取 Faiss / ANN 常见的 .fvecs 文件。
    每条向量格式：
        [int32 dim][float32 x dim]

    返回:
        ndarray, shape = (n, d), dtype=float32
    """
    if mmap:
        data = np.memmap(filename, dtype=np.int32, mode='r')
        dim = data[0]
        if dim <= 0:
            raise ValueError(f"Invalid dimension {dim} in file: {filename}")

        row_size = dim + 1
        if data.size % row_size != 0:
            raise ValueError(
                f"File size is not compatible with dim={dim}. "
                f"total_int32={data.size}, row_size={row_size}"
            )

        data = data.reshape(-1, row_size)
        dims = data[:, 0]
        if not np.all(dims == dim):
            raise ValueError("Inconsistent dimensions found in fvecs file.")

        # 这里要 view 成 float32
        vecs = data[:, 1:].view(np.float32)
        return vecs

    raw = np.fromfile(filename, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"Empty file: {filename}")

    dim = raw[0]
    if dim <= 0:
        raise ValueError(f"Invalid dimension {dim} in file: {filename}")

    row_size = dim + 1
    if raw.size % row_size != 0:
        raise ValueError(
            f"File size is not compatible with dim={dim}. "
            f"total_int32={raw.size}, row_size={row_size}"
        )

    raw = raw.reshape(-1, row_size)
    dims = raw[:, 0]
    if not np.all(dims == dim):
        raise ValueError("Inconsistent dimensions found in fvecs file.")

    vecs = raw[:, 1:].copy().view(np.float32)
    return vecs


def write_fvecs(filename, vecs):
    """
    将 float32 ndarray 写成 .fvecs 格式
    """
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim != 2:
        raise ValueError("vecs must be 2D")

    n, d = vecs.shape
    out = np.empty((n, d + 1), dtype=np.float32)
    out[:, 0] = d
    out[:, 1:] = vecs

    # 第一列需要按 int32 写入
    out_int = out.view(np.int32)
    out_int[:, 0] = d
    out_int.tofile(filename)


def fit_pca(
    train_vectors,
    out_model_path,
    n_components,
    whiten=False,
    svd_solver="randomized",
    random_state=42,
):
    """
    训练 PCA，并保存模型参数。
    """
    if train_vectors.ndim != 2:
        raise ValueError("train_vectors must be 2D")

    n, d = train_vectors.shape
    if n_components <= 0 or n_components > d:
        raise ValueError(
            f"n_components must be in [1, {d}], got {n_components}"
        )

    print(f"[INFO] Fitting PCA on {n} vectors, dim={d}, target_dim={n_components}")

    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver=svd_solver,
        random_state=random_state,
    )
    pca.fit(train_vectors)

    np.savez_compressed(
        out_model_path,
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        singular_values=pca.singular_values_.astype(np.float32),
        n_components=np.int32(n_components),
        original_dim=np.int32(d),
        whiten=np.int32(1 if whiten else 0),
    )

    print(f"[INFO] PCA model saved to: {out_model_path}")
    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"[INFO] Total explained variance ratio: {explained:.6f}")


def load_pca_model(model_path):
    """
    加载保存的 PCA 参数
    """
    model = np.load(model_path)

    result = {
        "mean": model["mean"],
        "components": model["components"],
        "explained_variance": model["explained_variance"],
        "explained_variance_ratio": model["explained_variance_ratio"],
        "singular_values": model["singular_values"],
        "n_components": int(model["n_components"]),
        "original_dim": int(model["original_dim"]),
        "whiten": bool(int(model["whiten"])),
    }
    return result


def apply_pca(vectors, pca_model):
    """
    直接使用保存好的 PCA 参数做变换，不重新训练。

    如果 whiten=False:
        Y = (X - mean) @ components.T

    如果 whiten=True:
        Y = ((X - mean) @ components.T) / sqrt(explained_variance)
    """
    X = np.asarray(vectors, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("vectors must be 2D")

    mean = pca_model["mean"]
    components = pca_model["components"]
    explained_variance = pca_model["explained_variance"]
    original_dim = pca_model["original_dim"]
    whiten = pca_model["whiten"]

    if X.shape[1] != original_dim:
        raise ValueError(
            f"Input dim mismatch: expected {original_dim}, got {X.shape[1]}"
        )

    X_centered = X - mean
    Y = X_centered @ components.T

    if whiten:
        eps = 1e-12
        Y = Y / np.sqrt(explained_variance + eps)

    return Y.astype(np.float32)


def cmd_fit(args):
    train_vecs = read_fvecs(args.train_fvecs, mmap=args.mmap)
    fit_pca(
        train_vectors=train_vecs,
        out_model_path=args.model_out,
        n_components=args.n_components,
        whiten=args.whiten,
        svd_solver=args.svd_solver,
        random_state=args.random_state,
    )


def cmd_transform(args):
    model = load_pca_model(args.model_in)
    vecs = read_fvecs(args.input_fvecs, mmap=args.mmap)
    transformed = apply_pca(vecs, model)

    if args.output_npy:
        np.save(args.output_npy, transformed)
        print(f"[INFO] Saved PCA vectors to NPY: {args.output_npy}")

    if args.output_fvecs:
        write_fvecs(args.output_fvecs, transformed)
        print(f"[INFO] Saved PCA vectors to FVECS: {args.output_fvecs}")

    if not args.output_npy and not args.output_fvecs:
        print("[WARN] No output file specified. Nothing was saved.")
        print("[INFO] Transformed shape:", transformed.shape)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and apply PCA for SIFT-1M .fvecs data"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fit
    p_fit = subparsers.add_parser("fit", help="Fit PCA and save model")
    p_fit.add_argument("--train-fvecs", required=True, help="Path to training .fvecs")
    p_fit.add_argument("--model-out", required=True, help="Output PCA model (.npz)")
    p_fit.add_argument("--n-components", type=int, required=True, help="Target PCA dim")
    p_fit.add_argument("--whiten", action="store_true", help="Enable whitening")
    p_fit.add_argument(
        "--svd-solver",
        default="randomized",
        choices=["auto", "full", "arpack", "randomized"],
        help="sklearn PCA svd_solver",
    )
    p_fit.add_argument("--random-state", type=int, default=42)
    p_fit.add_argument("--mmap", action="store_true", help="Use memmap to read fvecs")
    p_fit.set_defaults(func=cmd_fit)

    # transform
    p_trans = subparsers.add_parser(
        "transform", help="Load PCA model and transform vectors directly"
    )
    p_trans.add_argument("--model-in", required=True, help="Saved PCA model (.npz)")
    p_trans.add_argument("--input-fvecs", required=True, help="Input .fvecs to transform")
    p_trans.add_argument("--output-npy", help="Output transformed vectors as .npy")
    p_trans.add_argument("--output-fvecs", help="Output transformed vectors as .fvecs")
    p_trans.add_argument("--mmap", action="store_true", help="Use memmap to read fvecs")
    p_trans.set_defaults(func=cmd_transform)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()