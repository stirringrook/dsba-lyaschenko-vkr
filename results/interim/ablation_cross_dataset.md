# CREMA-D -> RAVDESS zero-shot (RH5)

Checkpoint trained on CREMA-D train split; evaluated on RAVDESS val split with no adaptation.

| run               | variant     |   ccc_V |   ccc_A |   CCC_VA |   F1_EXPR_macro |   F1_AU@0.5 |   P_MTL@0.5 |   trainable_params |
|:------------------|:------------|--------:|--------:|---------:|----------------:|------------:|------------:|-------------------:|
| crema_visual_only | visual_only |  0.7372 |  0.3984 |   0.5678 |          0.3907 |      0.0000 |      0.9585 |             179706 |
| crema_audio_only  | audio_only  |  0.0561 |  0.0859 |   0.0710 |          0.1631 |      0.0000 |      0.2341 |             142998 |
| crema_f1_concat   | f1_concat   |  0.6735 |  0.2732 |   0.4733 |          0.3496 |      0.0000 |      0.8230 |             995690 |
| crema_f2_blend    | f2_blend    |  0.7230 |  0.2887 |   0.5059 |          0.3607 |      0.0000 |      0.8666 |             322708 |
| crema_f3_gate     | f3_gate     |  0.6761 |  0.3808 |   0.5284 |          0.3649 |      0.0000 |      0.8933 |             624168 |
| crema_f4_xattn    | f4_xattn    |  0.6958 |  0.2743 |   0.4850 |          0.3714 |      0.0000 |      0.8564 |            1720470 |
| crema_f5_lmf      | f5_lmf      |  0.7028 |  0.3968 |   0.5498 |          0.3411 |      0.0000 |      0.8909 |             597914 |
