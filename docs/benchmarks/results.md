# Best Observed H100 Results

Maximum observed throughput per configuration, not a median across attempts. WER and suite time belong to the same selected run.

| Model | Precision | Batch | Best RTFx | Suite time | Mean WER | Attempts |
|---|---|---:|---:|---:|---:|---:|
| zipformer | fp32 | 1 | 715.9 | 793.37 s | 5.256% | 1 |
| zipformer | fp32 | 2 | 1,296.1 | 438.20 s | 5.254% | 1 |
| zipformer | fp32 | 4 | 2,308.0 | 246.08 s | 5.254% | 1 |
| zipformer | fp32 | 8 | 3,981.5 | 142.65 s | 5.249% | 1 |
| zipformer | fp32 | 16 | 6,158.5 | 92.22 s | 5.250% | 1 |
| zipformer | fp32 | 32 | 8,495.2 | 66.86 s | 5.247% | 1 |
| zipformer | fp32 | 64 | 10,646.3 | 53.35 s | 5.249% | 1 |
| zipformer | fp32 | 128 | 12,591.8 | 45.11 s | 5.247% | 1 |
| zipformer | fp16 | 1 | 766.3 | 741.19 s | 5.256% | 1 |
| zipformer | fp16 | 2 | 1,341.7 | 423.31 s | 5.251% | 1 |
| zipformer | fp16 | 4 | 2,388.6 | 237.78 s | 5.249% | 1 |
| zipformer | fp16 | 8 | 4,121.4 | 137.81 s | 5.250% | 1 |
| zipformer | fp16 | 16 | 6,861.6 | 82.77 s | 5.246% | 1 |
| zipformer | fp16 | 32 | 10,355.4 | 54.85 s | 5.244% | 1 |
| zipformer | fp16 | 64 | 14,154.1 | 40.13 s | 5.247% | 1 |
| zipformer | fp16 | 128 | 18,252.2 | 31.12 s | 5.243% | 2 |
| zipformer | fp16 | 256 | 20,242.2 | 28.06 s | 5.244% | 2 |
| zipformer | bf16 | 1 | 753.3 | 753.97 s | 5.250% | 1 |
| zipformer | bf16 | 2 | 1,120.8 | 506.73 s | 5.254% | 1 |
| zipformer | bf16 | 4 | 2,312.3 | 245.63 s | 5.246% | 1 |
| zipformer | bf16 | 8 | 3,961.1 | 143.38 s | 5.243% | 1 |
| zipformer | bf16 | 16 | 6,416.5 | 88.52 s | 5.247% | 1 |
| zipformer | bf16 | 32 | 9,898.9 | 57.38 s | 5.244% | 1 |
| zipformer | bf16 | 64 | 13,639.2 | 41.64 s | 5.246% | 1 |
| zipformer | bf16 | 128 | 17,359.0 | 32.72 s | 5.240% | 1 |
| zipformer | bf16 | 256 | 18,492.5 | 30.71 s | 5.250% | 1 |
| parakeet | fp32 | 1 | 656.8 | 864.79 s | 4.873% | 1 |
| parakeet | fp32 | 2 | 1,009.5 | 562.63 s | 4.870% | 1 |
| parakeet | fp32 | 4 | 1,965.1 | 289.02 s | 4.870% | 1 |
| parakeet | fp32 | 8 | 3,117.5 | 182.18 s | 4.869% | 1 |
| parakeet | fp32 | 16 | 4,417.5 | 128.57 s | 4.870% | 1 |
| parakeet | fp32 | 32 | 5,681.8 | 99.96 s | 4.877% | 1 |
| parakeet | fp32 | 64 | 6,661.6 | 85.26 s | 4.874% | 1 |
| parakeet | fp32 | 128 | 8,050.6 | 70.55 s | 4.876% | 1 |
| parakeet | fp16 | 1 | 825.1 | 688.31 s | 4.863% | 1 |
| parakeet | fp16 | 2 | 1,485.0 | 382.46 s | 4.881% | 1 |
| parakeet | fp16 | 4 | 2,606.2 | 217.92 s | 4.867% | 1 |
| parakeet | fp16 | 8 | 4,340.1 | 130.86 s | 4.876% | 1 |
| parakeet | fp16 | 16 | 6,367.0 | 89.20 s | 4.874% | 1 |
| parakeet | fp16 | 32 | 8,595.2 | 66.08 s | 4.881% | 1 |
| parakeet | fp16 | 64 | 10,605.4 | 53.55 s | 4.901% | 1 |
| parakeet | fp16 | 128 | 12,564.5 | 45.20 s | 4.876% | 2 |
| parakeet | fp16 | 256 | 13,561.8 | 41.88 s | 4.879% | 2 |
| parakeet | bf16 | 1 | 821.3 | 691.50 s | 4.881% | 1 |
| parakeet | bf16 | 2 | 1,477.8 | 384.31 s | 4.891% | 1 |
| parakeet | bf16 | 4 | 2,589.1 | 219.36 s | 4.873% | 1 |
| parakeet | bf16 | 8 | 4,146.2 | 136.98 s | 4.893% | 1 |
| parakeet | bf16 | 16 | 6,021.3 | 94.32 s | 4.873% | 1 |
| parakeet | bf16 | 32 | 8,289.9 | 68.51 s | 4.876% | 1 |
| parakeet | bf16 | 64 | 10,331.9 | 54.97 s | 4.886% | 1 |
| parakeet | bf16 | 128 | 12,263.6 | 46.31 s | 4.881% | 1 |
| parakeet | bf16 | 256 | 13,188.2 | 43.07 s | 4.893% | 1 |

Detailed measurements are retained locally, not included in Git. See [methodology](methodology.md).
