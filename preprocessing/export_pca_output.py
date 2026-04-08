import argparse
import os
import struct

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export PCA model .npz into binary files used by the C pipeline"
    )
    parser.add_argument(
        "--input",
        default="/home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/sift_pca_128.npz",
        help="Path to PCA model .npz",
    )
    parser.add_argument(
        "--outdir",
        default="/home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output",
        help="Output directory for exported .bin files",
    )
    parser.add_argument(
        "--prefix",
        default="pca",
        help="Output file prefix, e.g. pca -> pca_mean.bin",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model = np.load(args.input)
    print("keys:", model.files)

    required_keys = [
        "mean",
        "components",
        "explained_variance",
        "n_components",
        "original_dim",
        "whiten",
    ]
    for key in required_keys:
        if key not in model:
            raise KeyError(f"Missing key in npz: {key}")

    mean = model["mean"].astype(np.float32)
    components = model["components"].astype(np.float32)
    explained_variance = model["explained_variance"].astype(np.float32)
    n_components = int(model["n_components"])
    original_dim = int(model["original_dim"])
    whiten = int(model["whiten"])

    print("mean:", mean.shape, mean.dtype)
    print("components:", components.shape, components.dtype)
    print("explained_variance:", explained_variance.shape, explained_variance.dtype)
    print("n_components:", n_components)
    print("original_dim:", original_dim)
    print("whiten:", whiten)

    mean_path = os.path.join(args.outdir, f"{args.prefix}_mean.bin")
    components_path = os.path.join(args.outdir, f"{args.prefix}_components.bin")
    variance_path = os.path.join(args.outdir, f"{args.prefix}_explained_variance.bin")
    meta_path = os.path.join(args.outdir, f"{args.prefix}_meta.bin")

    with open(mean_path, "wb") as f:
        f.write(struct.pack("I", mean.shape[0]))
        f.write(mean.tobytes(order="C"))

    with open(components_path, "wb") as f:
        rows, cols = components.shape
        f.write(struct.pack("II", rows, cols))
        f.write(components.tobytes(order="C"))

    with open(variance_path, "wb") as f:
        f.write(struct.pack("I", explained_variance.shape[0]))
        f.write(explained_variance.tobytes(order="C"))

    with open(meta_path, "wb") as f:
        f.write(struct.pack("III", n_components, original_dim, whiten))

    print("export done")
    print("mean ->", mean_path)
    print("components ->", components_path)
    print("explained_variance ->", variance_path)
    print("meta ->", meta_path)


if __name__ == "__main__":
    main()
