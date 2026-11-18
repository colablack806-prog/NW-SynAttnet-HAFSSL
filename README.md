# HAFSSL / SynAttnNet: code, results and data

Code, saved results and data of the paper *Hybrid Attention-Enhanced Federated Semi-Supervised Learning for Network Traffic Classification* (under review).

```
data/          the three datasets used in the paper (see data/DATA_SOURCES.md)
experiments/   one folder per table: the script(s) and results/ with the saved outputs of the reported runs
tools/         evaluation and checking scripts
```

Each results zip holds the saved outputs of its runs: `results/all_runs.csv` (one row per run: dataset, method, seed, settings, accuracy, macro metrics and `tf_version`), the test-set predictions, per-run reports and the configuration file. All metrics in the paper are computed from the saved predictions.

## Setup

```
pip install -r requirements.txt
unzip data/ISCX-VPN2016.zip  -d data/ISCX-VPN2016
unzip data/10PopularApps.zip -d data/10PopularApps
```
For USTC-TFC2016, see `data/DATA_SOURCES.md`.

Run every command from the repository root. The scripts were run on Kaggle (NVIDIA Tesla T4) and write to `/kaggle/working` by default; set `OUT_DIR` (or `OUT_ROOT`, `BASE`) at the top of a script to write elsewhere. All settings are constants at the top of each script, with the values of the reported runs. `QUICK_TEST = True` runs a short check of the pipeline.

**Software version.** The `tf_version` column of every `all_runs.csv` records the TensorFlow version of each run. The four forms of the federated ablation (Table V) were all run with TensorFlow 2.21.0; the HAFSSL and FL-AECNN runs of Tables VI and VII used TensorFlow 2.20.0 (see Section V-D of the paper).

**Protocol shared by all scripts.** Exact duplicate flows are removed before splitting; the remaining flows are split once with seed 2026 into training, validation and test sets; seeds 0, 1, 2 (0-9 on USTC-TFC2016) fix the labeled flows, client partitions, initial weights and data order, and all compared methods receive the same ones; the final model is evaluated once on the test set.

## Tables and commands

### Table II: centralized comparison
```
python experiments/T02_centralized_comparison/01_centralized_synattnnet_R2.py   # SynAttnNet
python experiments/T02_centralized_comparison/02_centralized_AECNN_R2.py        # AECNN
python experiments/T02_centralized_comparison/03_centralized_baselines.py       # ADGCN, ByteSGAN, FLUIDS, VAE
```
Each run covers one dataset: set `DATASET` and the data path at the top (`data/ISCX-VPN2016/ISCX_5class_each_normalized_cuttedfloefeature.csv` or `data/10PopularApps/pcapdroid_10class_each_normalized_cuttedfloefeature.csv`). The baselines script reuses the split and labeled flows of the SynAttnNet run (`REFERENCE_DIR`, `AECNN_DIR`).
Results: `SynAttnNet_ISCX.zip`, `SynAttnNet_Apps10.zip`, `AECNN_ISCX.zip`, `AECNN_Apps10.zip`, `baselines_ISCX.zip`, `baselines_Apps10.zip`.

### Table IV: centralized ablation of the attention block
```
python experiments/T04_centralized_ablation/05_centralized_ablation_all_R2.py
```
Five forms (MHA->SE, SE->MHA, MHA only, SE only, no block) x 4 label budgets x 3 seeds on ISCX-VPN2016. Results: `SynAttnet-Ablation-All.zip` (60 runs).

### Table V: federated ablation of the attention block
```
python experiments/T05_federated_ablation/federated_ablation_R2.py   # 12 runs -> outputs/FED_ISCX_ablation
python experiments/T05_federated_ablation/table5_analysis.py         # Table V, paired tests and lost classes
```
Results: `ablation_TF2.21.zip` (the 12 runs of Table V; `unzip ablation_TF2.21.zip -d outputs` and run `table5_analysis.py` to recompute the table) and `earlier_run_TF2.20.zip` (the earlier run of the three alternative forms with TensorFlow 2.20, discussed in Section V-D).

### Table VI: federated baselines
```
python experiments/T06_federated_baselines/fluids_fed_vs_feat.py   # FLUIDS-fed, FEAT, and FEAT's CNN with plain FedAvg
python experiments/T06_federated_baselines/drfl.py                 # DRFL; SUPERVISED_FEDAVG_CONTROL = True gives the plain-FedAvg control
```
Results: `FLUIDvsFEAT_federated_baselines.zip`, `DRFL_federated_baselines.zip`, `Control_federated_baselines.zip`. In the DRFL zips, `papercnn` is DRFL's own CNN (Table VI) and `aecnn` runs DRFL with the FL-AECNN network instead. The HAFSSL and FL-AECNN rows of Table VI are the FedAvg runs of Table VII.

### Table VII: aggregators
```
python experiments/T07_aggregators/run_table7_2gpu.py      # two GPUs: FedAvg + FedProx, FedNova + MOON
python experiments/T07_aggregators/03_federated_table7_R2.py   # or all four aggregators on one GPU
```
Results: `Aggregators.zip` (`T7_gpu0`, `T7_gpu1`, `T7_combined`; 24 runs).

### Tables VIII and IX: extreme heterogeneity (alpha = 0.01)
```
python experiments/T08_T09_extreme_heterogeneity/04_federated_alpha001.py                  # FL-AECNN and HAFSSL (MODEL = "FL-AECNN" / "HAFSSL")
python experiments/T08_T09_extreme_heterogeneity/04b_federated_alpha001_HAFSSL-CA_ISCX.py   # HAFSSL-CA, ISCX-VPN2016
python experiments/T08_T09_extreme_heterogeneity/04c_federated_alpha001_HAFSSL-CA_Apps10.py # HAFSSL-CA, 10 Popular Apps
```
`04_federated_alpha001.py` is set for 10 Popular Apps; the ISCX-VPN2016 runs used the same code with `DATASET = "ISCX"`, the ISCX data path and labels. In `04b`, `MODEL = "HAFSSL"` is HAFSSL-CA (hybrid attention also in the classifier head, key dimension 16, head dropout 0.3; 461,013 parameters), which is why its output folder is named `..._HAFSSL`.
Results: `ISCX_FL-AECNN.zip`, `ISCX_HAFSSL.zip`, `ISCX_HAFSSL-CA.zip`, `Apps10_FL-AECNN_HAFSSL_HAFSSL-CA.zip` (HAFSSL-CA is the `clfatt` folder).

### Table X: other datasets and IID control
```
python experiments/T10_crossdataset_iid/05_federated_apps10_HAFSSL_vs_FLAECNN_R2.py   # 10 Popular Apps, non-IID
python experiments/T10_crossdataset_iid/05_federated_USTC_HAFSSL.py                   # USTC-TFC2016, HAFSSL, 10 seeds
python experiments/T10_crossdataset_iid/05_federated_USTC_FLAECNN.py                  # USTC-TFC2016, FL-AECNN, 10 seeds
python experiments/T10_crossdataset_iid/06_federated_IID_ISCX_Apps10_R2.py            # IID control, both datasets
```
Results: `Apps10_nonIID.zip`, `USTC_HAFSSL.zip`, `USTC_FL-AECNN.zip`, `IID_control_ISCX_Apps10.zip`.

### Table XI: computational and communication cost
```
python experiments/T11_cost/efficiency_worker.py
```
Results: `cost_HAFSSL_FLAECNN.zip`.

### Section V-B: ET-BERT on USTC-TFC2016
```
python experiments/V-B_ETBERT_USTC/08_centralized_USTC_L2000.py          # SynAttnNet, AECNN, AECNN-ft
python experiments/V-B_ETBERT_USTC/09_centralized_USTC_ETBERT_L2000.py   # ET-BERT (PyTorch)
```
The ET-BERT script clones https://github.com/linwhitehat/ET-BERT and needs the authors' pre-trained model (`MODEL_BIN`). Results: `CEN_USTC_L2000.zip`, `CEN_USTC_ETBERT_L2000.zip`.

## Tools
`tools/evaluate_all.py` recomputes the metrics from the saved predictions, `tools/partition_check.py` checks that compared runs share the same client partitions and labeled flows, and `tools/ustc_data_facts.py` gives the USTC-TFC2016 counts of Section V-A. Usage is described at the top of each script.
