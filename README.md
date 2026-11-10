# sglang-runtime

SGLang built from source for NVIDIA Blackwell (SM120) and Hopper (SM90), with FlashInfer and DeepGEMM from source and SM120 fixes not yet upstream.

## Image

| Tag | Built | Digest |
|---|---|---|
| `ghcr.io/infer-lab/sglang-runtime:20261108-7ef2921ac4f8` | 2026-11-08 | `sha256:8a7b0895b0563260dd5230eabe08f83dc25177813a6a3802530da83d0cbccfa6` |
| `ghcr.io/infer-lab/sglang-runtime:stable` | 2026-11-08 | `sha256:8a7b0895b0563260dd5230eabe08f83dc25177813a6a3802530da83d0cbccfa6` |

## Usage

The entrypoint activates the image's environment and runs the given command.

```sh
docker run --gpus all --ipc=host -p 30000:30000 -v /path/to/models:/models \
  ghcr.io/infer-lab/sglang-runtime:stable \
  python -m sglang.launch_server --model-path /models/<model> --tp 2 \
  --host 0.0.0.0 --port 30000
```

## Contents

### sglang

[`7ce26f621cb5`](https://github.com/lucifer1004/sglang/commit/7ce26f621cb5d10d065da4d90e7f7127a8757470): [sgl-project/sglang](https://github.com/sgl-project/sglang) at [`cbbc6b67f4`](https://github.com/sgl-project/sglang/commit/cbbc6b67f4)

plus:

- [sglang-torch-build-keep-rustflags](https://github.com/lucifer1004/sglang/tree/fix/torch-build-keep-rustflags): [Build] Keep CARGO_BUILD_RUSTFLAGS when torch-linked Rust extensions set RUSTFLAGS
- [sglang-chunked-host-pin](https://github.com/lucifer1004/sglang/tree/fix/chunked-host-pin-1108): [Bugfix] Pin the Qwen4-Exp PLE and DSv4.1 Engram host tables in 64 MiB chunks
- [sglang#42566](https://github.com/sgl-project/sglang/pull/42566): [Fix] quantization: Load block-FP8 MoE TP shards that split checkpoint blocks
- [sglang-stop-strings-after-reasoning](https://github.com/lucifer1004/sglang/tree/fix/stop-strings-after-reasoning): [Bugfix] Match stop strings only after the reasoning ends
- [sglang-qwen38-mtp-moe-configs](https://github.com/lucifer1004/sglang/tree/inferlab/qwen38-mtp-moe-configs-7224): [Perf] Tune Triton MoE configs for Qwen3.8-Flash-Next's MTP experts on RTX PRO 6000
- [sglang#49098](https://github.com/sgl-project/sglang/pull/49098): [Perf] [NVIDIA] dsa: Run FlashInfer's fused SM120 indexer for DSA top-k

### flashinfer

[`757780dcb9d6`](https://github.com/lucifer1004/flashinfer/commit/757780dcb9d6c12d8254970461db82cbab791c67): [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) at [`5d2c80277`](https://github.com/flashinfer-ai/flashinfer/commit/5d2c80277)

plus:

- [flashinfer#8740](https://github.com/flashinfer-ai/flashinfer/pull/8740): feat(sm120_dsa_indexer): DeepSeek-V4.1 decode in the paged form: strided page caches and candidate blocks

### DeepGEMM-sglang

[`9a464db6cdcb`](https://github.com/lucifer1004/DeepGEMM/commit/9a464db6cdcb86f6b43e0e1d4da5be42fdb01176): [sgl-project/DeepGEMM](https://github.com/sgl-project/DeepGEMM) at [`0f96f72`](https://github.com/sgl-project/DeepGEMM/commit/0f96f72)

plus:

- [sgl-deep-gemm-dist-name](https://github.com/lucifer1004/DeepGEMM/commit/d03fc66): build: publish the sglang-lineage dist as sgl-deep-gemm
- [deepgemm-setup-torch-library-ext](https://github.com/lucifer1004/DeepGEMM/tree/fix/setup-torch-library-ext-v026): build: setup.py builds the TORCH_LIBRARY _C that deep_gemm loads

## Serving

Each model was gated with the command below, run on the image's entrypoint with the model directory mounted; the parallelism and every setting shown are the ones that were measured.

### [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) — TP1

```sh
python3 -m sglang.launch_server --model-path <model> --tensor-parallel-size 1 --context-length 16384 --mem-fraction-static 0.85 --trust-remote-code --enable-cache-report
```

### [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) — TP2

```sh
python3 -m sglang.launch_server --model-path <model> --ple-offload-embedding --reasoning-parser qwen3 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-moe-runner-backend deep_gemm --tensor-parallel-size 2 --context-length 262144 --mem-fraction-static 0.85 --moe-runner-backend flashinfer_cutlass
```

## Validation

Gated on RTX PRO 6000 Blackwell (SM120) GPUs before publishing; scores are chat GSM8K (lm-eval, 1,319 questions, flexible-extract).

| Model | Configuration | Eval | Result |
|---|---|---|---|
| [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | TP1 | smoke | passed |
| [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | TP2 | smoke | passed |
| [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | TP2 | gsm8k-chat | 0.9757 (gate 0.96) |

Built and gated with InferLab 0.23.3.
