# Recorded experiments

`dnf-supervised-20261012` contains the completed supervised DNF search: 96 configurations, 12 seed confirmations, and five validation-selected outer refits. Of 113 trials, 110 completed and three triggered numerical guards. The queue ran from 2026-10-12 16:53:43 UTC to 21:20:52 UTC.

The locked winner is `t077-f0972bfbf3`: width 96, two root/history layers, history length 128, dropout 0.2, batch 64, AdamW learning rate 0.001, and no weight decay. Its five test AUROCs are 0.722491, 0.741136, 0.723208, 0.733816, and 0.742229: mean 0.732576 and sample standard deviation 0.009453. The three-seed validation mean was 0.657200. The fixed outer refit budget was 325 updates, selected before test evaluation.

Each outer run fitted all 11,977 train-plus-validation anchors and predicted all 702 masked test anchors. The five final checkpoints and prediction Parquets are copied here. JSON evidence remains byte-for-byte intact, including original lab paths and source/data digests. `catalog.json` maps final artifacts to their local paths. Prepared datasets and search checkpoints remain in the lab archive.

`references/relarena-20261012.csv` is the public results snapshot used for comparisons. Published scores are predominantly single-seed results; this experiment reports all five predeclared seeds rather than selecting its best test seed. Historical SSL pilots remain under `benchmarks/relarena` and the existing verification manifests.

Source snapshots and model checkpoints are reproducibility artifacts, excluded from Git and built wheels. They are not the installed RelFlow dependency.
