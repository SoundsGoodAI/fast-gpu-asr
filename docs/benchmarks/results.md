# Benchmark Results

156 complete configurations on A100, H200, B300. Both models use decoder beam 6, with one measured pass per configuration over seven English datasets. FP32 covers batches 1-128; FP16 and BF16 also cover 256.

[Methodology and hardware](methodology.md) | [CSV and plot reproduction](methodology.md#rebuild-plots) | [Measurements CSV](measurements.csv)

We reproduced the [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) English evaluation with fast-gpu-asr, using its datasets and WER scorer and reporting both **RTFx and WER**.

RTFx uses pooled real audio seconds divided by synchronized full-ASR seconds; padding, loading, file I/O, resampling, and scoring are excluded. Mean WER is the unweighted mean of seven upstream-rounded WERs. Precision labels retain exporter math defaults, including TF32 for FP32. Single-pass results do not estimate repeatability or establish that small differences are significant.

Dataset columns below use the cleaned AMI, Earnings22, GigaSpeech, and VoxPopuli sets. Earnings22 chunks are reassembled before scoring. All WER values are percentages; timing and WER come from the same pass.

## A100

### Throughput

| Model | Precision | Batch | RTFx | Suite time | Mean WER |
|---|---|---:|---:|---:|---:|
| Zipformer CR-CTC-transducer | FP16 | 1 | 589.0 | 964.35 s | 5.254% |
| Zipformer CR-CTC-transducer | FP16 | 2 | 974.7 | 582.68 s | 5.257% |
| Zipformer CR-CTC-transducer | FP16 | 4 | 1,594.0 | 356.30 s | 5.261% |
| Zipformer CR-CTC-transducer | FP16 | 8 | 2,525.2 | 224.91 s | 5.263% |
| Zipformer CR-CTC-transducer | FP16 | 16 | 3,843.8 | 147.76 s | 5.260% |
| Zipformer CR-CTC-transducer | FP16 | 32 | 5,516.7 | 102.95 s | 5.256% |
| Zipformer CR-CTC-transducer | FP16 | 64 | 7,387.0 | 76.89 s | 5.257% |
| Zipformer CR-CTC-transducer | FP16 | 128 | 9,093.0 | 62.46 s | 5.251% |
| Zipformer CR-CTC-transducer | FP16 | 256 | 10,298.5 | 55.15 s | 5.259% |
| Zipformer CR-CTC-transducer | BF16 | 1 | 587.5 | 966.78 s | 5.257% |
| Zipformer CR-CTC-transducer | BF16 | 2 | 956.5 | 593.78 s | 5.261% |
| Zipformer CR-CTC-transducer | BF16 | 4 | 1,557.6 | 364.64 s | 5.271% |
| Zipformer CR-CTC-transducer | BF16 | 8 | 2,470.9 | 229.85 s | 5.263% |
| Zipformer CR-CTC-transducer | BF16 | 16 | 3,731.5 | 152.20 s | 5.260% |
| Zipformer CR-CTC-transducer | BF16 | 32 | 5,288.2 | 107.40 s | 5.263% |
| Zipformer CR-CTC-transducer | BF16 | 64 | 6,990.0 | 81.25 s | 5.257% |
| Zipformer CR-CTC-transducer | BF16 | 128 | 8,659.6 | 65.59 s | 5.253% |
| Zipformer CR-CTC-transducer | BF16 | 256 | 9,794.3 | 57.99 s | 5.251% |
| Zipformer CR-CTC-transducer | FP32 | 1 | 532.7 | 1066.20 s | 5.251% |
| Zipformer CR-CTC-transducer | FP32 | 2 | 880.5 | 645.02 s | 5.259% |
| Zipformer CR-CTC-transducer | FP32 | 4 | 1,426.6 | 398.11 s | 5.264% |
| Zipformer CR-CTC-transducer | FP32 | 8 | 2,121.5 | 267.71 s | 5.264% |
| Zipformer CR-CTC-transducer | FP32 | 16 | 2,998.8 | 189.39 s | 5.263% |
| Zipformer CR-CTC-transducer | FP32 | 32 | 3,996.3 | 142.12 s | 5.257% |
| Zipformer CR-CTC-transducer | FP32 | 64 | 4,996.4 | 113.67 s | 5.257% |
| Zipformer CR-CTC-transducer | FP32 | 128 | 5,878.6 | 96.61 s | 5.259% |
| Parakeet V3 | FP16 | 1 | 560.8 | 1012.79 s | 4.824% |
| Parakeet V3 | FP16 | 2 | 913.9 | 621.47 s | 4.827% |
| Parakeet V3 | FP16 | 4 | 1,433.7 | 396.14 s | 4.817% |
| Parakeet V3 | FP16 | 8 | 2,208.3 | 257.19 s | 4.814% |
| Parakeet V3 | FP16 | 16 | 3,076.2 | 184.63 s | 4.831% |
| Parakeet V3 | FP16 | 32 | 4,156.3 | 136.65 s | 4.844% |
| Parakeet V3 | FP16 | 64 | 5,173.3 | 109.79 s | 4.829% |
| Parakeet V3 | FP16 | 128 | 5,931.9 | 95.75 s | 4.830% |
| Parakeet V3 | FP16 | 256 | 6,482.0 | 87.62 s | 4.816% |
| Parakeet V3 | BF16 | 1 | 567.9 | 1000.01 s | 4.806% |
| Parakeet V3 | BF16 | 2 | 908.5 | 625.16 s | 4.801% |
| Parakeet V3 | BF16 | 4 | 1,428.1 | 397.69 s | 4.816% |
| Parakeet V3 | BF16 | 8 | 2,172.5 | 261.43 s | 4.796% |
| Parakeet V3 | BF16 | 16 | 2,991.5 | 189.86 s | 4.800% |
| Parakeet V3 | BF16 | 32 | 4,121.5 | 137.80 s | 4.813% |
| Parakeet V3 | BF16 | 64 | 5,103.2 | 111.29 s | 4.830% |
| Parakeet V3 | BF16 | 128 | 5,759.1 | 98.62 s | 4.816% |
| Parakeet V3 | BF16 | 256 | 6,379.1 | 89.03 s | 4.793% |
| Parakeet V3 | FP32 | 1 | 426.1 | 1332.79 s | 4.839% |
| Parakeet V3 | FP32 | 2 | 668.3 | 849.80 s | 4.819% |
| Parakeet V3 | FP32 | 4 | 1,038.7 | 546.77 s | 4.831% |
| Parakeet V3 | FP32 | 8 | 1,473.8 | 385.36 s | 4.836% |
| Parakeet V3 | FP32 | 16 | 2,041.6 | 278.20 s | 4.833% |
| Parakeet V3 | FP32 | 32 | 2,610.8 | 217.54 s | 4.826% |
| Parakeet V3 | FP32 | 64 | 3,065.4 | 185.28 s | 4.827% |
| Parakeet V3 | FP32 | 128 | 3,541.0 | 160.39 s | 4.830% |

<details>
<summary>All per-dataset WERs</summary>

| Model | Precision | Batch | AMI | Earnings22 | GigaSpeech | LS Clean | LS Other | SPGISpeech | VoxPopuli |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zipformer CR-CTC-transducer | FP16 | 1 | 10.22 | 7.92 | 8.33 | 1.31 | 3.01 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | FP16 | 2 | 10.21 | 7.94 | 8.33 | 1.31 | 3.01 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP16 | 4 | 10.22 | 7.96 | 8.33 | 1.31 | 3.01 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP16 | 8 | 10.23 | 7.94 | 8.33 | 1.31 | 3.02 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | FP16 | 16 | 10.23 | 7.92 | 8.32 | 1.31 | 3.03 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | FP16 | 32 | 10.21 | 7.92 | 8.33 | 1.31 | 3.02 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP16 | 64 | 10.21 | 7.90 | 8.33 | 1.31 | 3.02 | 1.64 | 4.39 |
| Zipformer CR-CTC-transducer | FP16 | 128 | 10.21 | 7.89 | 8.33 | 1.31 | 3.03 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | FP16 | 256 | 10.21 | 7.92 | 8.32 | 1.31 | 3.03 | 1.64 | 4.38 |
| Zipformer CR-CTC-transducer | BF16 | 1 | 10.24 | 7.93 | 8.33 | 1.31 | 3.00 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | BF16 | 2 | 10.24 | 7.91 | 8.33 | 1.31 | 3.02 | 1.64 | 4.38 |
| Zipformer CR-CTC-transducer | BF16 | 4 | 10.23 | 8.00 | 8.34 | 1.31 | 3.02 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | BF16 | 8 | 10.25 | 7.93 | 8.34 | 1.31 | 3.03 | 1.64 | 4.34 |
| Zipformer CR-CTC-transducer | BF16 | 16 | 10.24 | 7.91 | 8.34 | 1.30 | 3.02 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | BF16 | 32 | 10.21 | 7.94 | 8.33 | 1.31 | 3.02 | 1.64 | 4.39 |
| Zipformer CR-CTC-transducer | BF16 | 64 | 10.24 | 7.90 | 8.33 | 1.31 | 3.01 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | BF16 | 128 | 10.22 | 7.89 | 8.33 | 1.31 | 3.02 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | BF16 | 256 | 10.20 | 7.90 | 8.33 | 1.31 | 3.02 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP32 | 1 | 10.22 | 7.90 | 8.33 | 1.31 | 3.01 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | FP32 | 2 | 10.22 | 7.94 | 8.33 | 1.31 | 3.01 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP32 | 4 | 10.22 | 7.96 | 8.33 | 1.31 | 3.01 | 1.64 | 4.38 |
| Zipformer CR-CTC-transducer | FP32 | 8 | 10.24 | 7.95 | 8.33 | 1.31 | 3.01 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | FP32 | 16 | 10.24 | 7.94 | 8.32 | 1.31 | 3.01 | 1.64 | 4.38 |
| Zipformer CR-CTC-transducer | FP32 | 32 | 10.22 | 7.92 | 8.32 | 1.31 | 3.02 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | FP32 | 64 | 10.22 | 7.92 | 8.32 | 1.31 | 3.02 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | FP32 | 128 | 10.23 | 7.92 | 8.32 | 1.31 | 3.02 | 1.64 | 4.37 |
| Parakeet V3 | FP16 | 1 | 8.95 | 5.90 | 7.76 | 1.45 | 3.08 | 3.56 | 3.07 |
| Parakeet V3 | FP16 | 2 | 8.96 | 5.90 | 7.76 | 1.47 | 3.08 | 3.56 | 3.06 |
| Parakeet V3 | FP16 | 4 | 8.98 | 5.85 | 7.75 | 1.45 | 3.07 | 3.56 | 3.06 |
| Parakeet V3 | FP16 | 8 | 8.95 | 5.84 | 7.75 | 1.45 | 3.08 | 3.56 | 3.07 |
| Parakeet V3 | FP16 | 16 | 8.92 | 5.98 | 7.76 | 1.45 | 3.07 | 3.56 | 3.08 |
| Parakeet V3 | FP16 | 32 | 8.99 | 5.98 | 7.75 | 1.46 | 3.08 | 3.56 | 3.09 |
| Parakeet V3 | FP16 | 64 | 8.98 | 5.90 | 7.75 | 1.45 | 3.09 | 3.56 | 3.07 |
| Parakeet V3 | FP16 | 128 | 8.94 | 5.96 | 7.75 | 1.45 | 3.08 | 3.56 | 3.07 |
| Parakeet V3 | FP16 | 256 | 8.97 | 5.83 | 7.75 | 1.46 | 3.07 | 3.56 | 3.07 |
| Parakeet V3 | BF16 | 1 | 8.94 | 5.77 | 7.75 | 1.45 | 3.09 | 3.56 | 3.08 |
| Parakeet V3 | BF16 | 2 | 8.92 | 5.78 | 7.75 | 1.47 | 3.09 | 3.55 | 3.05 |
| Parakeet V3 | BF16 | 4 | 8.91 | 5.87 | 7.76 | 1.46 | 3.09 | 3.55 | 3.07 |
| Parakeet V3 | BF16 | 8 | 8.92 | 5.73 | 7.76 | 1.46 | 3.09 | 3.55 | 3.06 |
| Parakeet V3 | BF16 | 16 | 8.94 | 5.74 | 7.76 | 1.46 | 3.08 | 3.55 | 3.07 |
| Parakeet V3 | BF16 | 32 | 8.91 | 5.85 | 7.76 | 1.46 | 3.09 | 3.55 | 3.07 |
| Parakeet V3 | BF16 | 64 | 8.92 | 5.98 | 7.75 | 1.46 | 3.08 | 3.55 | 3.07 |
| Parakeet V3 | BF16 | 128 | 8.92 | 5.86 | 7.75 | 1.46 | 3.09 | 3.56 | 3.07 |
| Parakeet V3 | BF16 | 256 | 8.94 | 5.75 | 7.73 | 1.47 | 3.08 | 3.55 | 3.03 |
| Parakeet V3 | FP32 | 1 | 8.97 | 5.97 | 7.75 | 1.45 | 3.09 | 3.56 | 3.08 |
| Parakeet V3 | FP32 | 2 | 8.96 | 5.83 | 7.76 | 1.45 | 3.09 | 3.56 | 3.08 |
| Parakeet V3 | FP32 | 4 | 8.96 | 5.94 | 7.75 | 1.45 | 3.08 | 3.56 | 3.08 |
| Parakeet V3 | FP32 | 8 | 8.96 | 5.96 | 7.75 | 1.45 | 3.09 | 3.56 | 3.08 |
| Parakeet V3 | FP32 | 16 | 8.93 | 6.01 | 7.74 | 1.45 | 3.08 | 3.56 | 3.06 |
| Parakeet V3 | FP32 | 32 | 8.91 | 5.99 | 7.74 | 1.45 | 3.08 | 3.56 | 3.05 |
| Parakeet V3 | FP32 | 64 | 8.93 | 5.98 | 7.74 | 1.45 | 3.08 | 3.56 | 3.05 |
| Parakeet V3 | FP32 | 128 | 8.96 | 5.93 | 7.75 | 1.45 | 3.08 | 3.56 | 3.08 |

</details>

## H200

### Throughput

| Model | Precision | Batch | RTFx | Suite time | Mean WER |
|---|---|---:|---:|---:|---:|
| Zipformer CR-CTC-transducer | FP16 | 1 | 578.2 | 982.29 s | 5.254% |
| Zipformer CR-CTC-transducer | FP16 | 2 | 971.1 | 584.86 s | 5.257% |
| Zipformer CR-CTC-transducer | FP16 | 4 | 1,654.7 | 343.24 s | 5.264% |
| Zipformer CR-CTC-transducer | FP16 | 8 | 2,747.7 | 206.71 s | 5.266% |
| Zipformer CR-CTC-transducer | FP16 | 16 | 4,430.5 | 128.19 s | 5.261% |
| Zipformer CR-CTC-transducer | FP16 | 32 | 6,989.0 | 81.26 s | 5.260% |
| Zipformer CR-CTC-transducer | FP16 | 64 | 10,384.6 | 54.69 s | 5.254% |
| Zipformer CR-CTC-transducer | FP16 | 128 | 14,304.1 | 39.71 s | 5.260% |
| Zipformer CR-CTC-transducer | FP16 | 256 | 18,100.2 | 31.38 s | 5.260% |
| Zipformer CR-CTC-transducer | BF16 | 1 | 561.2 | 1012.12 s | 5.259% |
| Zipformer CR-CTC-transducer | BF16 | 2 | 929.5 | 611.05 s | 5.264% |
| Zipformer CR-CTC-transducer | BF16 | 4 | 1,555.3 | 365.18 s | 5.256% |
| Zipformer CR-CTC-transducer | BF16 | 8 | 2,639.2 | 215.20 s | 5.260% |
| Zipformer CR-CTC-transducer | BF16 | 16 | 4,208.2 | 134.97 s | 5.269% |
| Zipformer CR-CTC-transducer | BF16 | 32 | 6,778.4 | 83.79 s | 5.271% |
| Zipformer CR-CTC-transducer | BF16 | 64 | 10,121.9 | 56.11 s | 5.260% |
| Zipformer CR-CTC-transducer | BF16 | 128 | 13,751.0 | 41.30 s | 5.260% |
| Zipformer CR-CTC-transducer | BF16 | 256 | 17,035.5 | 33.34 s | 5.254% |
| Zipformer CR-CTC-transducer | FP32 | 1 | 545.2 | 1041.74 s | 5.253% |
| Zipformer CR-CTC-transducer | FP32 | 2 | 940.2 | 604.05 s | 5.256% |
| Zipformer CR-CTC-transducer | FP32 | 4 | 1,575.0 | 360.60 s | 5.261% |
| Zipformer CR-CTC-transducer | FP32 | 8 | 2,577.9 | 220.32 s | 5.260% |
| Zipformer CR-CTC-transducer | FP32 | 16 | 4,191.1 | 135.51 s | 5.261% |
| Zipformer CR-CTC-transducer | FP32 | 32 | 6,244.0 | 90.96 s | 5.259% |
| Zipformer CR-CTC-transducer | FP32 | 64 | 8,632.1 | 65.80 s | 5.259% |
| Zipformer CR-CTC-transducer | FP32 | 128 | 11,077.1 | 51.27 s | 5.259% |
| Parakeet V3 | FP16 | 1 | 593.6 | 956.80 s | 4.803% |
| Parakeet V3 | FP16 | 2 | 1,028.5 | 552.20 s | 4.821% |
| Parakeet V3 | FP16 | 4 | 1,717.9 | 330.60 s | 4.819% |
| Parakeet V3 | FP16 | 8 | 2,795.5 | 203.17 s | 4.807% |
| Parakeet V3 | FP16 | 16 | 4,118.6 | 137.90 s | 4.809% |
| Parakeet V3 | FP16 | 32 | 6,097.5 | 93.15 s | 4.816% |
| Parakeet V3 | FP16 | 64 | 8,331.7 | 68.17 s | 4.816% |
| Parakeet V3 | FP16 | 128 | 10,412.1 | 54.55 s | 4.814% |
| Parakeet V3 | FP16 | 256 | 12,352.7 | 45.98 s | 4.804% |
| Parakeet V3 | BF16 | 1 | 591.3 | 960.52 s | 4.811% |
| Parakeet V3 | BF16 | 2 | 1,001.9 | 566.90 s | 4.797% |
| Parakeet V3 | BF16 | 4 | 1,686.7 | 336.73 s | 4.819% |
| Parakeet V3 | BF16 | 8 | 2,668.6 | 212.83 s | 4.801% |
| Parakeet V3 | BF16 | 16 | 4,023.5 | 141.16 s | 4.806% |
| Parakeet V3 | BF16 | 32 | 6,003.5 | 94.60 s | 4.796% |
| Parakeet V3 | BF16 | 64 | 8,065.3 | 70.42 s | 4.830% |
| Parakeet V3 | BF16 | 128 | 10,353.9 | 54.85 s | 4.831% |
| Parakeet V3 | BF16 | 256 | 12,397.9 | 45.81 s | 4.820% |
| Parakeet V3 | FP32 | 1 | 507.1 | 1119.96 s | 4.844% |
| Parakeet V3 | FP32 | 2 | 799.1 | 710.75 s | 4.839% |
| Parakeet V3 | FP32 | 4 | 1,435.9 | 395.54 s | 4.843% |
| Parakeet V3 | FP32 | 8 | 2,285.1 | 248.55 s | 4.834% |
| Parakeet V3 | FP32 | 16 | 3,292.9 | 172.48 s | 4.836% |
| Parakeet V3 | FP32 | 32 | 4,600.3 | 123.46 s | 4.833% |
| Parakeet V3 | FP32 | 64 | 5,680.0 | 99.99 s | 4.819% |
| Parakeet V3 | FP32 | 128 | 7,371.8 | 77.04 s | 4.841% |

<details>
<summary>All per-dataset WERs</summary>

| Model | Precision | Batch | AMI | Earnings22 | GigaSpeech | LS Clean | LS Other | SPGISpeech | VoxPopuli |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zipformer CR-CTC-transducer | FP16 | 1 | 10.23 | 7.90 | 8.33 | 1.31 | 3.01 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP16 | 2 | 10.23 | 7.92 | 8.33 | 1.31 | 3.01 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP16 | 4 | 10.23 | 7.98 | 8.33 | 1.31 | 3.01 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | FP16 | 8 | 10.24 | 7.95 | 8.33 | 1.31 | 3.02 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | FP16 | 16 | 10.24 | 7.95 | 8.32 | 1.31 | 3.01 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP16 | 32 | 10.22 | 7.94 | 8.32 | 1.31 | 3.02 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | FP16 | 64 | 10.21 | 7.92 | 8.32 | 1.31 | 3.02 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP16 | 128 | 10.22 | 7.93 | 8.32 | 1.31 | 3.02 | 1.64 | 4.38 |
| Zipformer CR-CTC-transducer | FP16 | 256 | 10.23 | 7.92 | 8.32 | 1.31 | 3.02 | 1.64 | 4.38 |
| Zipformer CR-CTC-transducer | BF16 | 1 | 10.22 | 7.93 | 8.34 | 1.31 | 3.02 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | BF16 | 2 | 10.24 | 7.96 | 8.33 | 1.31 | 3.02 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | BF16 | 4 | 10.23 | 7.90 | 8.33 | 1.31 | 3.01 | 1.65 | 4.36 |
| Zipformer CR-CTC-transducer | BF16 | 8 | 10.22 | 7.93 | 8.33 | 1.31 | 3.02 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | BF16 | 16 | 10.26 | 7.94 | 8.34 | 1.31 | 3.02 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | BF16 | 32 | 10.23 | 7.98 | 8.33 | 1.31 | 3.02 | 1.64 | 4.39 |
| Zipformer CR-CTC-transducer | BF16 | 64 | 10.23 | 7.92 | 8.33 | 1.31 | 3.03 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | BF16 | 128 | 10.22 | 7.95 | 8.33 | 1.31 | 3.02 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | BF16 | 256 | 10.21 | 7.92 | 8.33 | 1.31 | 3.02 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | FP32 | 1 | 10.22 | 7.91 | 8.33 | 1.31 | 3.01 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | FP32 | 2 | 10.22 | 7.92 | 8.33 | 1.31 | 3.01 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP32 | 4 | 10.22 | 7.96 | 8.33 | 1.31 | 3.01 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP32 | 8 | 10.22 | 7.94 | 8.33 | 1.31 | 3.01 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | FP32 | 16 | 10.23 | 7.94 | 8.32 | 1.31 | 3.01 | 1.64 | 4.38 |
| Zipformer CR-CTC-transducer | FP32 | 32 | 10.21 | 7.94 | 8.32 | 1.31 | 3.02 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | FP32 | 64 | 10.22 | 7.92 | 8.32 | 1.31 | 3.02 | 1.64 | 4.38 |
| Zipformer CR-CTC-transducer | FP32 | 128 | 10.23 | 7.92 | 8.32 | 1.31 | 3.02 | 1.64 | 4.37 |
| Parakeet V3 | FP16 | 1 | 8.97 | 5.74 | 7.75 | 1.45 | 3.08 | 3.56 | 3.07 |
| Parakeet V3 | FP16 | 2 | 8.94 | 5.92 | 7.76 | 1.45 | 3.07 | 3.56 | 3.05 |
| Parakeet V3 | FP16 | 4 | 8.93 | 5.91 | 7.76 | 1.45 | 3.07 | 3.56 | 3.05 |
| Parakeet V3 | FP16 | 8 | 8.93 | 5.83 | 7.76 | 1.46 | 3.07 | 3.56 | 3.04 |
| Parakeet V3 | FP16 | 16 | 8.95 | 5.82 | 7.76 | 1.45 | 3.06 | 3.56 | 3.06 |
| Parakeet V3 | FP16 | 32 | 8.95 | 5.84 | 7.76 | 1.46 | 3.07 | 3.56 | 3.07 |
| Parakeet V3 | FP16 | 64 | 8.95 | 5.84 | 7.75 | 1.45 | 3.08 | 3.56 | 3.08 |
| Parakeet V3 | FP16 | 128 | 8.98 | 5.81 | 7.75 | 1.45 | 3.07 | 3.56 | 3.08 |
| Parakeet V3 | FP16 | 256 | 8.95 | 5.79 | 7.75 | 1.46 | 3.07 | 3.56 | 3.05 |
| Parakeet V3 | BF16 | 1 | 8.88 | 5.87 | 7.76 | 1.46 | 3.08 | 3.55 | 3.08 |
| Parakeet V3 | BF16 | 2 | 8.92 | 5.76 | 7.75 | 1.46 | 3.07 | 3.56 | 3.06 |
| Parakeet V3 | BF16 | 4 | 8.98 | 5.83 | 7.75 | 1.45 | 3.08 | 3.56 | 3.08 |
| Parakeet V3 | BF16 | 8 | 8.90 | 5.79 | 7.75 | 1.46 | 3.09 | 3.56 | 3.06 |
| Parakeet V3 | BF16 | 16 | 8.88 | 5.85 | 7.75 | 1.46 | 3.08 | 3.55 | 3.07 |
| Parakeet V3 | BF16 | 32 | 8.90 | 5.77 | 7.74 | 1.45 | 3.08 | 3.56 | 3.07 |
| Parakeet V3 | BF16 | 64 | 8.91 | 6.01 | 7.75 | 1.45 | 3.07 | 3.56 | 3.06 |
| Parakeet V3 | BF16 | 128 | 8.90 | 6.00 | 7.76 | 1.45 | 3.09 | 3.56 | 3.06 |
| Parakeet V3 | BF16 | 256 | 8.94 | 5.88 | 7.75 | 1.46 | 3.08 | 3.56 | 3.07 |
| Parakeet V3 | FP32 | 1 | 8.97 | 6.01 | 7.76 | 1.45 | 3.08 | 3.56 | 3.08 |
| Parakeet V3 | FP32 | 2 | 8.96 | 5.98 | 7.76 | 1.45 | 3.09 | 3.56 | 3.07 |
| Parakeet V3 | FP32 | 4 | 8.95 | 6.02 | 7.75 | 1.45 | 3.09 | 3.56 | 3.08 |
| Parakeet V3 | FP32 | 8 | 8.95 | 5.96 | 7.75 | 1.45 | 3.09 | 3.56 | 3.08 |
| Parakeet V3 | FP32 | 16 | 8.97 | 5.96 | 7.75 | 1.45 | 3.09 | 3.56 | 3.07 |
| Parakeet V3 | FP32 | 32 | 8.96 | 5.94 | 7.75 | 1.45 | 3.09 | 3.56 | 3.08 |
| Parakeet V3 | FP32 | 64 | 8.96 | 5.84 | 7.75 | 1.45 | 3.09 | 3.56 | 3.08 |
| Parakeet V3 | FP32 | 128 | 8.97 | 6.00 | 7.75 | 1.45 | 3.08 | 3.56 | 3.08 |

</details>

## B300

### Throughput

| Model | Precision | Batch | RTFx | Suite time | Mean WER |
|---|---|---:|---:|---:|---:|
| Zipformer CR-CTC-transducer | FP16 | 1 | 879.8 | 645.57 s | 5.257% |
| Zipformer CR-CTC-transducer | FP16 | 2 | 1,606.7 | 353.49 s | 5.256% |
| Zipformer CR-CTC-transducer | FP16 | 4 | 2,708.9 | 209.66 s | 5.261% |
| Zipformer CR-CTC-transducer | FP16 | 8 | 4,403.2 | 128.99 s | 5.264% |
| Zipformer CR-CTC-transducer | FP16 | 16 | 6,853.2 | 82.87 s | 5.261% |
| Zipformer CR-CTC-transducer | FP16 | 32 | 10,308.0 | 55.10 s | 5.257% |
| Zipformer CR-CTC-transducer | FP16 | 64 | 14,752.0 | 38.50 s | 5.259% |
| Zipformer CR-CTC-transducer | FP16 | 128 | 20,068.3 | 28.30 s | 5.261% |
| Zipformer CR-CTC-transducer | FP16 | 256 | 25,108.6 | 22.62 s | 5.261% |
| Zipformer CR-CTC-transducer | BF16 | 1 | 869.8 | 652.97 s | 5.256% |
| Zipformer CR-CTC-transducer | BF16 | 2 | 1,606.3 | 353.58 s | 5.256% |
| Zipformer CR-CTC-transducer | BF16 | 4 | 2,720.4 | 208.78 s | 5.259% |
| Zipformer CR-CTC-transducer | BF16 | 8 | 4,420.6 | 128.48 s | 5.263% |
| Zipformer CR-CTC-transducer | BF16 | 16 | 6,854.9 | 82.85 s | 5.257% |
| Zipformer CR-CTC-transducer | BF16 | 32 | 10,309.4 | 55.09 s | 5.264% |
| Zipformer CR-CTC-transducer | BF16 | 64 | 14,776.5 | 38.44 s | 5.259% |
| Zipformer CR-CTC-transducer | BF16 | 128 | 20,147.6 | 28.19 s | 5.256% |
| Zipformer CR-CTC-transducer | BF16 | 256 | 25,269.3 | 22.48 s | 5.250% |
| Zipformer CR-CTC-transducer | FP32 | 1 | 765.1 | 742.29 s | 5.253% |
| Zipformer CR-CTC-transducer | FP32 | 2 | 1,438.3 | 394.89 s | 5.260% |
| Zipformer CR-CTC-transducer | FP32 | 4 | 2,473.6 | 229.60 s | 5.261% |
| Zipformer CR-CTC-transducer | FP32 | 8 | 4,036.4 | 140.71 s | 5.263% |
| Zipformer CR-CTC-transducer | FP32 | 16 | 6,112.4 | 92.92 s | 5.264% |
| Zipformer CR-CTC-transducer | FP32 | 32 | 9,022.0 | 62.95 s | 5.256% |
| Zipformer CR-CTC-transducer | FP32 | 64 | 12,572.3 | 45.18 s | 5.257% |
| Zipformer CR-CTC-transducer | FP32 | 128 | 16,423.9 | 34.58 s | 5.260% |
| Parakeet V3 | FP16 | 1 | 897.5 | 632.82 s | 4.814% |
| Parakeet V3 | FP16 | 2 | 1,495.0 | 379.90 s | 4.817% |
| Parakeet V3 | FP16 | 4 | 2,548.0 | 222.90 s | 4.823% |
| Parakeet V3 | FP16 | 8 | 4,093.7 | 138.74 s | 4.819% |
| Parakeet V3 | FP16 | 16 | 6,074.9 | 93.49 s | 4.806% |
| Parakeet V3 | FP16 | 32 | 9,020.6 | 62.96 s | 4.799% |
| Parakeet V3 | FP16 | 64 | 12,440.2 | 45.66 s | 4.807% |
| Parakeet V3 | FP16 | 128 | 15,774.4 | 36.00 s | 4.814% |
| Parakeet V3 | FP16 | 256 | 19,398.7 | 29.28 s | 4.810% |
| Parakeet V3 | BF16 | 1 | 911.4 | 623.15 s | 4.829% |
| Parakeet V3 | BF16 | 2 | 1,526.2 | 372.13 s | 4.827% |
| Parakeet V3 | BF16 | 4 | 2,514.6 | 225.86 s | 4.824% |
| Parakeet V3 | BF16 | 8 | 4,049.9 | 140.24 s | 4.833% |
| Parakeet V3 | BF16 | 16 | 6,091.1 | 93.24 s | 4.831% |
| Parakeet V3 | BF16 | 32 | 9,099.8 | 62.41 s | 4.821% |
| Parakeet V3 | BF16 | 64 | 12,421.8 | 45.72 s | 4.821% |
| Parakeet V3 | BF16 | 128 | 15,739.9 | 36.08 s | 4.820% |
| Parakeet V3 | BF16 | 256 | 19,580.4 | 29.01 s | 4.816% |
| Parakeet V3 | FP32 | 1 | 721.0 | 787.76 s | 4.819% |
| Parakeet V3 | FP32 | 2 | 1,285.6 | 441.78 s | 4.826% |
| Parakeet V3 | FP32 | 4 | 2,161.9 | 262.72 s | 4.821% |
| Parakeet V3 | FP32 | 8 | 3,409.1 | 166.60 s | 4.821% |
| Parakeet V3 | FP32 | 16 | 5,138.4 | 110.53 s | 4.823% |
| Parakeet V3 | FP32 | 32 | 7,153.2 | 79.40 s | 4.821% |
| Parakeet V3 | FP32 | 64 | 8,395.4 | 67.65 s | 4.826% |
| Parakeet V3 | FP32 | 128 | 9,951.3 | 57.07 s | 4.824% |

<details>
<summary>All per-dataset WERs</summary>

| Model | Precision | Batch | AMI | Earnings22 | GigaSpeech | LS Clean | LS Other | SPGISpeech | VoxPopuli |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zipformer CR-CTC-transducer | FP16 | 1 | 10.24 | 7.91 | 8.33 | 1.31 | 3.01 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP16 | 2 | 10.23 | 7.92 | 8.32 | 1.31 | 3.02 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | FP16 | 4 | 10.23 | 7.97 | 8.32 | 1.31 | 3.01 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | FP16 | 8 | 10.24 | 7.96 | 8.33 | 1.31 | 3.01 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP16 | 16 | 10.24 | 7.94 | 8.33 | 1.30 | 3.02 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP16 | 32 | 10.22 | 7.93 | 8.32 | 1.31 | 3.02 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP16 | 64 | 10.22 | 7.91 | 8.32 | 1.31 | 3.03 | 1.64 | 4.38 |
| Zipformer CR-CTC-transducer | FP16 | 128 | 10.23 | 7.94 | 8.32 | 1.31 | 3.02 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | FP16 | 256 | 10.23 | 7.93 | 8.32 | 1.31 | 3.02 | 1.64 | 4.38 |
| Zipformer CR-CTC-transducer | BF16 | 1 | 10.23 | 7.92 | 8.33 | 1.31 | 3.02 | 1.64 | 4.34 |
| Zipformer CR-CTC-transducer | BF16 | 2 | 10.23 | 7.90 | 8.34 | 1.31 | 3.02 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | BF16 | 4 | 10.25 | 7.94 | 8.33 | 1.31 | 3.02 | 1.64 | 4.32 |
| Zipformer CR-CTC-transducer | BF16 | 8 | 10.23 | 7.94 | 8.33 | 1.31 | 3.02 | 1.65 | 4.36 |
| Zipformer CR-CTC-transducer | BF16 | 16 | 10.23 | 7.92 | 8.33 | 1.31 | 3.02 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | BF16 | 32 | 10.24 | 7.94 | 8.33 | 1.31 | 3.03 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | BF16 | 64 | 10.23 | 7.91 | 8.33 | 1.31 | 3.03 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | BF16 | 128 | 10.20 | 7.93 | 8.33 | 1.31 | 3.03 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | BF16 | 256 | 10.19 | 7.94 | 8.32 | 1.31 | 3.02 | 1.64 | 4.33 |
| Zipformer CR-CTC-transducer | FP32 | 1 | 10.23 | 7.90 | 8.33 | 1.31 | 3.01 | 1.64 | 4.35 |
| Zipformer CR-CTC-transducer | FP32 | 2 | 10.24 | 7.93 | 8.33 | 1.31 | 3.01 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP32 | 4 | 10.23 | 7.96 | 8.32 | 1.31 | 3.01 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP32 | 8 | 10.24 | 7.95 | 8.32 | 1.31 | 3.01 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | FP32 | 16 | 10.25 | 7.94 | 8.32 | 1.31 | 3.01 | 1.64 | 4.38 |
| Zipformer CR-CTC-transducer | FP32 | 32 | 10.22 | 7.92 | 8.32 | 1.31 | 3.02 | 1.64 | 4.36 |
| Zipformer CR-CTC-transducer | FP32 | 64 | 10.22 | 7.92 | 8.32 | 1.31 | 3.02 | 1.64 | 4.37 |
| Zipformer CR-CTC-transducer | FP32 | 128 | 10.23 | 7.92 | 8.32 | 1.31 | 3.02 | 1.64 | 4.38 |
| Parakeet V3 | FP16 | 1 | 8.98 | 5.81 | 7.76 | 1.45 | 3.07 | 3.56 | 3.07 |
| Parakeet V3 | FP16 | 2 | 8.97 | 5.83 | 7.75 | 1.45 | 3.08 | 3.56 | 3.08 |
| Parakeet V3 | FP16 | 4 | 8.96 | 5.91 | 7.75 | 1.46 | 3.07 | 3.56 | 3.05 |
| Parakeet V3 | FP16 | 8 | 8.92 | 5.90 | 7.76 | 1.46 | 3.08 | 3.56 | 3.05 |
| Parakeet V3 | FP16 | 16 | 8.96 | 5.79 | 7.75 | 1.46 | 3.07 | 3.56 | 3.05 |
| Parakeet V3 | FP16 | 32 | 8.96 | 5.74 | 7.75 | 1.45 | 3.08 | 3.56 | 3.05 |
| Parakeet V3 | FP16 | 64 | 8.92 | 5.82 | 7.76 | 1.45 | 3.08 | 3.56 | 3.06 |
| Parakeet V3 | FP16 | 128 | 8.98 | 5.81 | 7.76 | 1.46 | 3.08 | 3.56 | 3.05 |
| Parakeet V3 | FP16 | 256 | 8.94 | 5.81 | 7.75 | 1.45 | 3.08 | 3.56 | 3.08 |
| Parakeet V3 | BF16 | 1 | 8.93 | 5.98 | 7.75 | 1.44 | 3.09 | 3.56 | 3.05 |
| Parakeet V3 | BF16 | 2 | 8.94 | 5.95 | 7.75 | 1.45 | 3.09 | 3.56 | 3.05 |
| Parakeet V3 | BF16 | 4 | 8.93 | 5.91 | 7.75 | 1.46 | 3.08 | 3.56 | 3.08 |
| Parakeet V3 | BF16 | 8 | 8.95 | 5.94 | 7.76 | 1.46 | 3.08 | 3.56 | 3.08 |
| Parakeet V3 | BF16 | 16 | 8.93 | 5.94 | 7.75 | 1.46 | 3.09 | 3.56 | 3.09 |
| Parakeet V3 | BF16 | 32 | 8.93 | 5.90 | 7.75 | 1.45 | 3.09 | 3.56 | 3.07 |
| Parakeet V3 | BF16 | 64 | 8.92 | 5.91 | 7.74 | 1.46 | 3.07 | 3.56 | 3.09 |
| Parakeet V3 | BF16 | 128 | 8.92 | 5.90 | 7.74 | 1.46 | 3.08 | 3.56 | 3.08 |
| Parakeet V3 | BF16 | 256 | 8.95 | 5.82 | 7.75 | 1.47 | 3.09 | 3.56 | 3.07 |
| Parakeet V3 | FP32 | 1 | 8.96 | 5.83 | 7.75 | 1.45 | 3.09 | 3.56 | 3.09 |
| Parakeet V3 | FP32 | 2 | 8.97 | 5.87 | 7.76 | 1.45 | 3.09 | 3.56 | 3.08 |
| Parakeet V3 | FP32 | 4 | 8.99 | 5.83 | 7.76 | 1.45 | 3.08 | 3.56 | 3.08 |
| Parakeet V3 | FP32 | 8 | 8.98 | 5.83 | 7.76 | 1.45 | 3.09 | 3.56 | 3.08 |
| Parakeet V3 | FP32 | 16 | 8.98 | 5.87 | 7.75 | 1.45 | 3.09 | 3.56 | 3.06 |
| Parakeet V3 | FP32 | 32 | 8.98 | 5.87 | 7.75 | 1.45 | 3.08 | 3.56 | 3.06 |
| Parakeet V3 | FP32 | 64 | 8.98 | 5.88 | 7.75 | 1.45 | 3.09 | 3.56 | 3.07 |
| Parakeet V3 | FP32 | 128 | 8.98 | 5.88 | 7.75 | 1.45 | 3.09 | 3.56 | 3.06 |

</details>
