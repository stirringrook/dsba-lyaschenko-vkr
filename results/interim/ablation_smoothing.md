# Per-video smoothing ablation (CREMA-D)

Paper A's smoothing grid (sigma, delta) applied post-hoc at evaluation time to V/A regressions and EXPR probabilities. AU is NOT smoothed (cell 66 of Paper A confirmed it hurts AU F1). P_MTL here is the three-task variant used throughout the interim: CCC_V + CCC_A + F1_EXPR.

| variant           |   ccc_V_base |   ccc_A_base |   F1_EXPR_base |   P_MTL_base |   ccc_V_sm |   ccc_A_sm |   F1_EXPR_sm |   P_MTL_sm |   delta_P_MTL |   sigma_VA |   delta_VA |   sigma_EXPR |   delta_EXPR |
|:------------------|-------------:|-------------:|---------------:|-------------:|-----------:|-----------:|-------------:|-----------:|--------------:|-----------:|-----------:|-------------:|-------------:|
| crema_visual_only |       0.6821 |       0.2989 |         0.3799 |       1.3609 |     0.7514 |     0.3505 |       0.4453 |     1.5471 |        0.1863 |         10 |         50 |           50 |           50 |
| crema_f1_concat   |       0.7401 |       0.4525 |         0.4389 |       1.6315 |     0.8280 |     0.5411 |       0.5221 |     1.8911 |        0.2596 |     100000 |         50 |          100 |           50 |
| crema_f3_gate     |       0.7352 |       0.5141 |         0.4270 |       1.6763 |     0.8095 |     0.5848 |       0.4956 |     1.8899 |        0.2136 |     100000 |         50 |          500 |           10 |
| crema_f4_xattn    |       0.8333 |       0.5390 |         0.4886 |       1.8609 |     0.8533 |     0.5576 |       0.5012 |     1.9121 |        0.0512 |     100000 |         50 |          100 |           50 |
