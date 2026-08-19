# onnxruntime-jetson

Prebuilt **ONNX Runtime GPU wheels (CUDAExecutionProvider) for aarch64 / NVIDIA
Jetson**, built on GitHub's native ARM64 runners instead of on the device.

PyPI's `onnxruntime-gpu` ships no aarch64 wheels, and the NVIDIA-affiliated
`pypi.jetson-ai-lab.io` index only carries Python 3.10 (cp310) GPU wheels built
on Ubuntu 24.04. This repo fills the gap with wheels that match the most common
robot userland:

| Property | Value |
| --- | --- |
| Platform | linux_aarch64 |
| Python ABI | cp312 |
| Build distro | Ubuntu 22.04 → glibc 2.35 (matches L4T R36.x userland) |
| CUDA toolkit (build) | 12.6 (sbsa), fallback 12.8 |
| cuDNN (build) | 9.3.0.75 (sbsa) |
| CUDA architectures | sm_87 (Jetson Orin) only |
| ONNX Runtime | 1.27.0 (pinned; other versions via manual dispatch) |

## Using a built wheel

Download the wheel from the Releases page, then on the Jetson:

```sh
uv pip install --python /path/to/venv/bin/python \
  onnxruntime_gpu-1.27.0-cp312-cp312-linux_aarch64.whl
```

Verify the CUDA provider actually initializes (GPU is needed only at runtime):

```python
import onnxruntime as ort
s = ort.InferenceSession("your_model.onnx",
                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(s.get_providers())
```

## Building

Push a `v*` tag or run the workflow manually with an `ort_version` input.
The build does not need a GPU: only the CUDA toolkit and cuDNN libraries are
required at link time.

ONNX Runtime is MIT-licensed Microsoft open source; the wheel here is built
unmodified from the official `v1.27.0` tag.
