# Aff-Wild2 Stage 1 reproduction summary


## Raw (no smoothing) vs. Paper A cell 60 "aligned" targets

| backbone | CCC_V (ours / paper) | CCC_A | F1_EXPR | P_MTL (ours / paper) | delta_P_MTL | within +/-0.015? |
| --- | --- | --- | --- | --- | --- | --- |
| `enet_b0_8_va_mtl` | 0.4640 / 0.4433 | 0.4211 / 0.3422 | 0.3369 / 0.5040 | 1.2591 / 1.2896 | -0.0305 | NO |
| `mbf_va_mtl` | 0.4697 / 0.4503 | 0.4339 / 0.2870 | 0.2913 / 0.4891 | 1.2217 / 1.2264 | -0.0047 | yes |

## Smoothing grid — best (sigma, delta) per backbone

| backbone | sigma* | delta* | P_MTL smoothed | delta_over_raw |
| --- | --- | --- | --- | --- |
| `enet_b0_8_va_mtl` | 500 | 10 | 1.3688 | +0.1097 |
| `mbf_va_mtl` | 1000.0 | 10 | 1.3039 | +0.0822 |
