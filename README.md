# HAFSSL — Hybrid Attention-Enhanced Federated Semi-Supervised Learning for Network Traffic Classification

Code for the manuscript submitted to *IEEE Transactions on Artificial Intelligence*.

This repository contains the implementation of **SynAttnNet**, a traffic-aware hybrid attention autoencoder–CNN model, and its federated semi-supervised extension **HAFSSL**. Only the proposed models are included; see [Baselines](#baselines).

---

## Files

### Centralized experiments

| File | Reproduces |
|---|---|
| `01_centralized_comparison.py` | Table II — proposed-model rows |
| `02_ablation_centralized.py` | Table III — attention ordering ablation (`VARIANT` switch) |

### Federated experiments — IID and non-IID (Tables V, VI)

| File | Reproduces | Partition |
|---|---|---|
| `03_federated_hafssl_iid.py` | Tables V, VI — IID rows | equal contiguous shards |
| `04_federated_hafssl_noniid.py` | Tables V, VI — non-IID rows | Dirichlet α = 0.5 |

### Federated experiments — attention ablation (Table IV)

| File | Attention configuration | Function name |
|---|---|---|
| `05_ablation_federated_MHA_SE.py` | MHA → SE (proposed) | `hybrid_attention_block` |
| `06_ablation_federated_se_only.py` | SE only | `se_attention_block` |
| `07_ablation_federated_mha_only.py` | MHA only | `mha_attention_block` |
| `08_ablation_federated_se_mha.py` | SE → MHA (reversed) | `se_mha_attention_block` |

All four run at K = 10, α = 0.5, 500 labels, 50 rounds. Each function name states exactly what the block contains, so the four variants are distinguishable at a glance. `se_mha` and `mha_se` contain identical layers in a different order and therefore have identical parameter counts; every script prints `count_params()` so this can be verified directly rather than taken on trust.

### Federated experiments — aggregation strategies (Table VII)

| File | Aggregator | Hyperparameters |
|---|---|---|
| `09_aggregator_fednova.py` | FedNova | normalization by local steps τ_k |
| `10_aggregator_fedavg.py` | FedAvg | — |
| `11_aggregator_fedprox.py` | FedProx | μ = 0.1 (proximal penalty on Stage 2) |
| `12_aggregator_moon.py` | MOON | μ = 5.0, τ = 0.5 (model-contrastive loss on Stage 2) |

FedProx and MOON modify the **local training objective**; aggregation stays weighted FedAvg. FedNova modifies the **server aggregation** rule. References: FedProx (Li et al., MLSys 2020), FedNova (Wang et al., NeurIPS 2020), MOON (Li et al., CVPR 2021).

### Scalability and extreme heterogeneity (Table VIII)

| File | Setting |
|---|---|
| `13_scalability_k50.py` | K = 50, α = 0.5, ISCX 5-class, 50 rounds |
| `14_extreme_heterogeneity_alpha001.py` | K = 10, α = 0.01, ISCX 5-class, 100 rounds |

At α = 0.01 most clients hold flows from a single service class. The partition routine retries until every client has at least 5 samples and warns if it cannot; clients falling below the 50-sample training floor are skipped and recorded with zero accuracy for that round.

### Cross-dataset evaluation (Table IX)

| File | Dataset |
|---|---|
| `15_cross_dataset_ustc.py` | USTC-TFC2016, 20 classes, K = 10, α = 0.5 |

This script consumes the first 128 16-bit payload words per flow, normalized to [0, 1] — **not** the 66 precomputed flow-statistical features used by the ISCX scripts. The cross-dataset evaluation therefore changes both the dataset and the input representation.

---

## Per-script configuration

Scripts differ in more than the variable under study. The table below records what each one actually runs, so that any comparison across tables can be checked against it. Values are read from the `CONFIG` block of each file and are also written to that run's `*_final_summary.csv`.

| Script | Dataset | K | α | Labels/client | Epochs | Rounds | Attn. in encoder | Attn. in head | Stage 1.5 | Aggregation |
|---|---|---|---|---|---|---|---|---|---|---|
| `03_` | ISCX 5 | 10 | IID | varies | 5 | 50 | yes | no | yes | unweighted mean |
| `04_` | ISCX 5 | 10 | 0.5 | varies | 5 | 50 | yes | no | yes | unweighted mean |
| `13_` | ISCX 5 | 50 | 0.5 | 500 | 5 | 50 | yes | yes | yes | weighted (n_k/n) |
| `14_` | ISCX 5 | 10 | 0.01 | 1000 | 3 | 100 | yes | yes | yes | weighted (n_k/n) |
| `15_` | USTC 20 | 10 | 0.5 | 2000 | 5 | 50 | yes | no | yes | weighted (n_k/n) |

**`labelnum` is a per-client budget, divided evenly across classes.** A value of 1000 on the 5-class data means 200 labeled samples per class *on each client*, so the federation-wide labeled total scales with K. This matters when comparing K = 10 against K = 50 at the same `labelnum`.

**Stage 1.5** is a short encoder fine-tuning pass (classifier head frozen, Adam at lr 1e-4, 2 epochs) run between the unsupervised autoencoder stage and full classifier training.

Attention dropout is 0.1 in the ISCX scripts and 0.05 in `15_`. Classifier-head dropout is 0.3 throughout.

---

## Table X — parameter sensitivity analysis

There is no separate script for Table X. Every row was produced by running `01_centralized_comparison.py` with **three constants changed and nothing else touched**.

| Constant in the script | Symbol in Table X | Values swept | Default elsewhere |
|---|---|---|---|
| `latent_dim` | d_latent | 16, 32, 64 | 39 |
| `NUM_HEADS` | H | 2, 4, 8 | 4 |
| `SE_RATIO` | r | 4, 8, 16 | 16 |
| `labelnum` | Labels | 500, 1000, 4000 | — |

That is 3 × 3 × 3 = 27 attention configurations per label regime, 81 runs in total. Table X reports only four rows per regime (best, 2nd best, worst, 2nd worst); the complete 81-row grid is in `results/sensitivity_table10.csv`.

**Held fixed across all 81 runs:** key dimension 32, dropout, Adam optimizer, batch size, epochs per stage, the attention ordering (MHA then SE), the random seed, and the same ISCX 5-class CSV with the same train/test split. Only the four values above vary, so any difference in accuracy is attributable to them.

**What the grid shows.** Latent dimension is the binding constraint: d_latent = 16 appears in almost every worst-case row at every label budget. Attention capacity has to scale with supervision — at 500 labels the strongest configurations use H = 2 and the weakest use H = 8, while at 1000 labels H = 8 becomes the best setting. The SE reduction ratio is the least sensitive of the three; values of 4, 8 and 16 all appear in both best and worst rows. Overall spread narrows as labels increase (1.5 points at 500 labels, 1.3 at 1000, 0.5 at 4000), so the model is most sensitive to these choices exactly where labels are scarcest.

---

## Data sources

Both flow-feature datasets are preprocessed CSV files released by Wang et al. in the repository accompanying their work on federated semi-supervised traffic classification.

**Source repository:** [HGW-TC-Experimental-code](https://github.com/PrinceXuan12138/HGW-TC-Experimental-code) by PrinceXuan12138

| Dataset | Classes | Direct download |
|---|---|---|
| ISCX-VPN2016 | 5 | [ISCX_5class_each_normalized_cuttedflowfeature.csv](https://github.com/PrinceXuan12138/HGW-TC-Experimental-code/releases/download/v1.0.0/ISCX_5class_each_normalized_cuttedflowfeature.csv) |
| 10 Popular Apps | 10 | [pcapdroid_10class_each_normalized_cuttedflowfeature.csv](https://github.com/PrinceXuan12138/HGW-TC-Experimental-code/releases/download/v1.0.0/pcapdroid_10class_each_normalized_cuttedflowfeature.csv) |

Both files contain flow-level statistical features already extracted and normalized, so no packet capture processing is required to run this code.

### USTC-TFC2016 preprocessing

`15_` loads six preprocessed `.npy` files (`x_payload_{train,valid,test}.npy`, `y_{train,valid,test}.npy`) containing space-separated 4-character hex tokens per flow. These are not produced by any script in this repository; a preprocessing script covering payload extraction, the train/validation/test split and hex tokenization still needs to be added before Table IX is independently reproducible.

### Raw datasets

| Dataset | Availability |
|---|---|
| ISCX-VPN2016 | [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/vpn.html) |
| USTC-TFC2016 | [yungshenglu/USTC-TFC2016](https://github.com/yungshenglu/USTC-TFC2016) (originally via [echowei/DeepTraffic](https://github.com/echowei/DeepTraffic)) |

---

## Environment and setup

The scripts were developed across two environments. Before running, update the dataset path and platform-specific settings at the top of each file.

| Scripts | Environment | Data access |
|---|---|---|
| `01`–`04`, `09`–`11` | Google Colab | Google Drive mount / `gdown` download |
| `05`–`08`, `12`–`15` | Kaggle | Kaggle input dataset path |

**For the Kaggle scripts**, change `KAGGLE_USERNAME`, `DATASET_PATH` (or `BASE_PATH`) and `CHECKPOINT_DATASET` to your own Kaggle account and uploaded dataset. **For the Colab scripts**, change the Drive file ID or point `dataset_path` at a local copy of the CSV.

Tested with TensorFlow 2.x, scikit-learn, pandas, numpy, matplotlib, seaborn. The Kaggle scripts additionally require `GPUtil`, `psutil` and the `kaggle` CLI, installed by the first line of each script.

### Checkpoint / resume behaviour

The Kaggle scripts save a checkpoint after every federated round and push it to a Kaggle Dataset, so a run interrupted by a session timeout can resume from the last completed round. Set `FORCE_FRESH_START = True` to discard any existing checkpoint and retrain from round 1.

The checkpoint location is derived from `MODEL_NAME` and the auto-generated folder name, which in turn includes epochs, batch size, round count and α. Changing any of those starts a new run from round 1 rather than resuming an old one — this is intentional, and is the simplest way to begin a clean experiment.

**Note on directory naming:** the folder `Models/AECNNmodel/` and the internal Keras layer names `encoder`, `decoder` and `AEcnn` are unchanged throughout. `load_model()` resolves layers by name, so renaming them would make existing checkpoints unloadable and would stop models saved by one script from loading in another. Despite the folder name, it stores **HAFSSL** client models.

---

## How to run

### Table II — centralized comparison

Set `labelnum` to 1000 / 2000 / 3000 / 4000 and run `01_centralized_comparison.py`. For the 10-class dataset, change the CSV filename and update the `LABELS` list.

### Table III — centralized ablation

Set `VARIANT` to one of `se_only`, `mha_only`, `se_mha`, `mha_se` and `labelnum` to 1000 or 3000, then run `02_ablation_centralized.py`. Each run appends a row to `ablation_table3.csv`, so after eight runs that file contains the full ablation table. Do not edit the attention block itself — the switch keeps the residual connection, layer normalizations and feed-forward sub-layer identical across all four variants, so any difference is attributable to the attention configuration alone.

### Table IV — federated ablation

Run each of `05`–`08` once. Model files and figures are prefixed with the variant name, so the four runs do not overwrite one another.

### Tables V and VI — federated IID and non-IID

Run `03_federated_hafssl_iid.py` for the IID rows and `04_federated_hafssl_noniid.py` for the non-IID rows. Set `labelnum` to 500 / 800 / 1000 for the label budget and switch the CSV path and `LABELS` list for the 10-class dataset.

In both cases each client first trains the hybrid-attention autoencoder on its unlabeled flows, then fine-tunes the classifier on its labeled subset; encoder and classifier weights are aggregated at the server.

### Table VII — aggregation strategies

Run each of `09`–`12` once. Every output file is prefixed with the model name (`HAFSSL-FedNova`, `HAFSSL-FedAvg`, `HAFSSL_FedProx`, `HAFSSL_MOON_10client`) so results from different aggregators remain separable.

`12_aggregator_moon.py` additionally records the contrastive loss per round in its convergence CSV and produces a dedicated `*_05_contrastive_loss.png` figure. The per-class confusion matrix from this run is the evidence for the VoIP class collapse discussed in Appendix C — a failure invisible at the global-accuracy level, which is why per-class reporting is included alongside aggregate metrics.

### Table VIII — scalability and extreme heterogeneity

Run `13_` for the K = 50 row and `14_` for the α = 0.01 row. `14_` uses 100 rounds rather than 50; state this in the table caption, since it differs from every other federated run.

Visualizations in `13_` are sized for 50 clients: per-client accuracy is drawn as a mean ± standard-deviation band rather than 50 individual lines, and final per-client metrics use horizontal bars.

### Table IX — cross-dataset evaluation

Run `15_` once. Point `BASE_PATH` at the directory containing the six USTC `.npy` files.

### Table X — sensitivity grid

See the dedicated section above. No script; the grid is a parameter sweep of `01_centralized_comparison.py`.

---

## Model configuration

Default architecture, matching Table I of the manuscript:

| Parameter | Value |
|---|---|
| Input features | 66 (ISCX) / 128 payload words (USTC) |
| Latent dimension | 39 |
| Attention heads | 4 |
| Key dimension | 32 |
| SE reduction ratio | 16 |
| Dropout | 0.1 |
| Optimizer | Adam |
| Local epochs | 5 |
| Global rounds | 50 |
| Clients (K) | 10 |
| Dirichlet concentration | α = 0.5 |

Training proceeds in two sequential stages: unsupervised autoencoder reconstruction on unlabeled flows, followed by supervised fine-tuning of the encoder together with the classifier on the limited labeled subset. Most scripts include an additional short encoder-only fine-tuning step with the classifier head frozen, between the two stages.

Individual scripts deviate from the defaults above; see the per-script configuration table. The values actually used are set in the `CONFIG` block at the top of each file and are recorded in the summary CSV each run produces.

---

## Outputs

Each federated script writes to a `result/` directory:

| File | Contents |
|---|---|
| `*_convergence.csv` | per-round global accuracy, precision, recall, F1 |
| `*_performance.csv` | per-client, per-round metrics and full classification reports |
| `*_monitoring.csv` | GPU / CPU / memory utilisation and wall-clock time per stage |
| `*_final_summary.csv` | one-row summary: configuration, final metrics, parameter count, inference latency |
| `result/figures/*.png` | non-IID distribution, convergence, per-client accuracy, confusion matrix, complexity, dashboard |

Confusion matrices (`*_06_confusion.png`) are computed from the aggregated global model, not from any individual client.

---

## Baselines

This repository contains only the proposed models, SynAttnNet and HAFSSL. No baseline implementation is included.

Results for AECNN, ADGCN, FLUIDS, ByteSGAN and VAE in Table II are quoted from the original publications and were not retrained in this work.

FL-AECNN, the federated baseline compared against HAFSSL in Tables V–IX, follows the design of Wang et al. [10]; its reference implementation is available in the authors' repository linked under **Data sources** and is not redistributed here.

---

## Status

**Included:** centralized experiments (Tables II, III); federated IID and non-IID comparisons (Tables V, VI); the four-variant federated attention ablation (Table IV); the four-aggregator comparison (Table VII); scalability and extreme heterogeneity (Table VIII); the USTC-TFC2016 cross-dataset evaluation (Table IX); the sensitivity grid recipe (Table X).

**Still to be added:**

| Item | Covers |
|---|---|
| `16_efficiency.py` | Table XI, Appendix A — parameter counts, communication per round, inference latency |
| `17_failure_analysis.py` | Appendix C — MOON VoIP collapse, α = 0.01 degenerate pattern, WhatsApp/Snapchat confusion |
| `18_make_figures.py` | Figs. 4, 5 — per-class precision / recall / F1 bar charts |
| `ustc_preprocess.py` | USTC-TFC2016 payload extraction and split, currently undocumented |
| `results/sensitivity_table10.csv` | the full 81-run grid behind Table X |
| `requirements.txt` | pinned dependencies (`pip freeze`); the TensorFlow version affects reproducibility |
| `.gitignore` | keeps `.keras`, `.h5`, data CSVs and `__pycache__` out of the repository |
| `LICENSE` | MIT or the institution's preferred licence |

Table XII (Appendix B) is a consolidation of results already reported in Tables V–IX rather than a separate experiment, and is assembled from the `*_final_summary.csv` files.

---

## Acknowledgment and dataset citation

The preprocessed datasets used here were released by Wang et al. Please cite their work when using these files:

    @article{wang2024network,
      title={Network traffic classification based on federated semi-supervised learning},
      author={Wang, ZiXuan and Li, ZeYi and Fu, Mengyi and Ye, YingChun and Wang, Pan},
      journal={Journal of Systems Architecture},
      volume={149},
      pages={103091},
      year={2024},
      publisher={Elsevier}
    }

USTC-TFC2016 was released by Wang et al. (2017):

    @inproceedings{wang2017malware,
      title={Malware traffic classification using convolutional neural network for representation learning},
      author={Wang, Wei and Zhu, Ming and Zeng, Xuewen and Ye, Xiaozhou and Sheng, Yiqiang},
      booktitle={2017 International Conference on Information Networking (ICOIN)},
      pages={712--717},
      year={2017},
      organization={IEEE}
    }

---

## Citation

    @article{hafssl,
      title={Hybrid Attention-Enhanced Federated Semi-Supervised Learning for Network Traffic Classification},
      journal={IEEE Transactions on Artificial Intelligence},
      year={2026}
    }
