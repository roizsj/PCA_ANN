import numpy as np
import struct

model = np.load("/home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/sift_pca_128.npz")
print("keys:", model.files)

mean = model["mean"].astype(np.float32)                         # [128]
components = model["components"].astype(np.float32)            # [128, 128]
explained_variance = model["explained_variance"].astype(np.float32)  # [128]
n_components = int(model["n_components"])
original_dim = int(model["original_dim"])
whiten = int(model["whiten"])

print("mean:", mean.shape, mean.dtype)
print("components:", components.shape, components.dtype)
print("explained_variance:", explained_variance.shape, explained_variance.dtype)
print("n_components:", n_components)
print("original_dim:", original_dim)
print("whiten:", whiten)

# 1) mean.bin
with open("/home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/pca_mean.bin", "wb") as f:
    f.write(struct.pack("I", mean.shape[0]))
    f.write(mean.tobytes(order="C"))

# 2) components.bin
with open("/home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/pca_components.bin", "wb") as f:
    rows, cols = components.shape
    f.write(struct.pack("II", rows, cols))
    f.write(components.tobytes(order="C"))

# 3) explained_variance.bin
with open("/home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/pca_explained_variance.bin", "wb") as f:
    f.write(struct.pack("I", explained_variance.shape[0]))
    f.write(explained_variance.tobytes(order="C"))

# 4) meta.bin
with open("/home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/pca_meta.bin", "wb") as f:
    f.write(struct.pack("III", n_components, original_dim, whiten))

print("export done")