# EMI §D — Multi-seed paired comparison (K=6, seeds=[42, 0, 1, 2, 3, 4])

Each variant trained K=6 times with different seeds. Per-seed deltas, mean ± s.d., paired-bootstrap 95 % CI on the mean delta (B=10,000), Wilcoxon signed-rank p-value (two-sided).

| Comparison | per-seed Δ | mean Δ ± s.d. | 95 % CI on mean Δ | sign | Wilcoxon p |
| --- | --- | --- | --- | ---: | ---: |
| xattn − text/mean | [+0.0191, +0.0149, +0.0090, +0.0226, +0.0105, +0.0126] | +0.0148 ± 0.0052 | [+0.0112, +0.0187] | 6/6 | 0.0312 |
| concat_mlp − text/mean | [+0.0173, +0.0168, +0.0110, +0.0178, +0.0155, +0.0184] | +0.0161 ± 0.0027 | [+0.0139, +0.0178] | 6/6 | 0.0312 |
| gated − text/mean | [-0.0026, +0.0062, -0.0047, +0.0101, +0.0050, -0.0003] | +0.0023 ± 0.0057 | [-0.0018, +0.0065] | 3/6 | 0.4375 |

Per-seed point estimates per variant (val mean Pearson):

| Variant | seed 42 | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean ± s.d. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `text/mean` | 0.4295 | 0.4289 | 0.4334 | 0.4280 | 0.4290 | 0.4275 | 0.4294 ± 0.0021 |
| `concat_mlp` | 0.4469 | 0.4457 | 0.4444 | 0.4458 | 0.4444 | 0.4459 | 0.4455 ± 0.0010 |
| `gated` | 0.4270 | 0.4350 | 0.4286 | 0.4381 | 0.4339 | 0.4271 | 0.4316 ± 0.0047 |
| `xattn` | 0.4486 | 0.4438 | 0.4424 | 0.4506 | 0.4395 | 0.4400 | 0.4441 ± 0.0046 |
