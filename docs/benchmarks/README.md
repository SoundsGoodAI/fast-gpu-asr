# H100 Benchmark Results

| Zipformer (beam 6) | Parakeet V3 (beam 6) |
|---|---|
| ![Zipformer fp32](zipformer-fp32.svg) | ![Parakeet fp32](parakeet-fp32.svg) |
| ![Zipformer fp16](zipformer-fp16.svg) | ![Parakeet fp16](parakeet-fp16.svg) |
| ![Zipformer bf16](zipformer-bf16.svg) | ![Parakeet bf16](parakeet-bf16.svg) |

Best-observed FP16 results at batch capacity 256 (not medians):

| Model | Batch | Precision | RTFx | Suite time | Mean WER |
|---|---:|---|---:|---:|---:|
| [zipformer](results.md) | 256 | FP16 | 20,242.2 | 28.06 s | 5.244% |
| [parakeet](results.md) | 256 | FP16 | 13,561.8 | 41.88 s | 4.879% |

[Complete results](results.md) | [Methodology and reproduction](methodology.md)
