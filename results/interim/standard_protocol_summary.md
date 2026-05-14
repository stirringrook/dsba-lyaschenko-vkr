# Standard-protocol summary across all interim runs

Clip-level accuracy and macro / weighted-F1 over the native CREMA-D / RAVDESS label spaces. These are the *same* checkpoints as the per-frame Aff-Wild2-aligned table in `summary_crema.md`; only the evaluation protocol changes.

## Label space: `cremad6`

| run | n_clips | accuracy | F1 macro | F1 weighted |
| --- | --- | --- | --- | --- |
| crema_audio_only | 1147 | 0.5222 | 0.5085 | 0.5064 |
| crema_f0_grid | 1147 | 0.7367 | 0.7364 | 0.7358 |
| crema_f1_concat | 1147 | 0.6966 | 0.6917 | 0.6926 |
| crema_f1_concat_mbf | 1147 | 0.6443 | 0.6368 | 0.6385 |
| crema_f2_blend | 1147 | 0.6792 | 0.6761 | 0.6768 |
| crema_f3_gate | 1147 | 0.6661 | 0.6618 | 0.6631 |
| crema_f4_xattn | 1147 | 0.6652 | 0.6646 | 0.6652 |
| crema_f5_lmf | 1147 | 0.6670 | 0.6597 | 0.6618 |
| crema_visual_only | 1147 | 0.6051 | 0.5949 | 0.5959 |

## Label space: `ravdess7`

| run | n_clips | accuracy | F1 macro | F1 weighted |
| --- | --- | --- | --- | --- |
| crema_audio_only_on_ravdess | 600 | 0.2183 | 0.1871 | 0.1962 |
| crema_f1_concat_on_ravdess | 600 | 0.4817 | 0.4454 | 0.4413 |
| crema_f2_blend_on_ravdess | 600 | 0.4833 | 0.4397 | 0.4356 |
| crema_f3_gate_on_ravdess | 600 | 0.5083 | 0.4493 | 0.4451 |
| crema_f4_xattn_on_ravdess | 600 | 0.4617 | 0.4280 | 0.4285 |
| crema_f5_lmf_on_ravdess | 600 | 0.4733 | 0.4171 | 0.4124 |
| crema_visual_only_on_ravdess | 600 | 0.5650 | 0.4906 | 0.4965 |
| ravdess_f1_concat | 600 | 0.6333 | 0.5960 | 0.5980 |
| ravdess_visual_only | 600 | 0.6217 | 0.5872 | 0.5865 |

## Label space: `ravdess8`

| run | n_clips | accuracy | F1 macro | F1 weighted |
| --- | --- | --- | --- | --- |
| crema_audio_only_on_ravdess | 600 | 0.2033 | 0.1768 | 0.1601 |
| crema_f1_concat_on_ravdess | 600 | 0.4667 | 0.4121 | 0.4020 |
| crema_f2_blend_on_ravdess | 600 | 0.4700 | 0.4093 | 0.3982 |
| crema_f3_gate_on_ravdess | 600 | 0.4917 | 0.4114 | 0.4032 |
| crema_f4_xattn_on_ravdess | 600 | 0.4333 | 0.3759 | 0.3711 |
| crema_f5_lmf_on_ravdess | 600 | 0.4633 | 0.3931 | 0.3812 |
| crema_visual_only_on_ravdess | 600 | 0.5367 | 0.4443 | 0.4273 |
| ravdess_f1_concat | 600 | 0.5700 | 0.4897 | 0.4976 |
| ravdess_visual_only | 600 | 0.5700 | 0.4933 | 0.4987 |

