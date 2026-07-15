# Custom / ImageWAM FLUX.2 4B LIBERO pilot

用途：ImageWAM custom backend 的 FLUX.2 4B + LIBERO pilot 入口。

依赖：

```text
upstreams/ImageWAM/
models/custom/imagewam/flux2_klein_4b_libero/
models/custom/imagewam/flux2/
data/custom/fastwam/libero-fastwam/
```

先准备资产：

```bash
make prepare-imagewam-upstream
make download-imagewam-artifacts
make download-imagewam-flux2-base
make download-custom-fastwam-libero-dataset
```

metadata smoke：

```bash
IMAGEWAM_MODE=metadata-smoke IMAGEWAM_REQUIRE_CUDA=0 \
bash experiments/custom/imagewam_flux2_4b_libero_pilot/launch.sh
```

pilot：

```bash
bash experiments/custom/imagewam_flux2_4b_libero_pilot/launch.sh
```
