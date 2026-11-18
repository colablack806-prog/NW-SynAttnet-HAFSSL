# Data

The three datasets in the versions used in the paper. Unzip them before running any script (from the repository root):

```
unzip data/ISCX-VPN2016.zip  -d data/ISCX-VPN2016
unzip data/10PopularApps.zip -d data/10PopularApps
unzip data/USTC-TFC2016.zip  -d data/USTC-TFC2016
```
If USTC-TFC2016 is split into parts (`USTC-TFC2016.zip` with `USTC-TFC2016.z01`, `.z02`, ...), join them first:
```
zip -s 0 data/USTC-TFC2016.zip --out data/USTC-TFC2016_full.zip
unzip data/USTC-TFC2016_full.zip -d data/USTC-TFC2016
```

| File | Content | Source |
|---|---|---|
| `ISCX-VPN2016.zip` | `ISCX_5class_each_normalized_cuttedfloefeature.csv`: 100,000 flows, 5 classes (Chat, Email, File, Streaming, VoIP), 66 min-max scaled flow features | ISCX VPN-nonVPN 2016, NonVPN-PCAP subset (Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/vpn.html), in the version provided by ref. [10] of the paper, with classes balanced to 20,000 flows by replication |
| `10PopularApps.zip` | `pcapdroid_10class_each_normalized_cuttedfloefeature.csv`: 180,000 flows, 10 applications (18,000 each), 66 flow features | ref. [10] of the paper |
| `USTC-TFC2016.zip` | `x_payload_{train,valid,test}.npy` (payload as hexadecimal tokens) and `y_{train,valid,test}.npy`: 97,115 flows, 20 classes (5,000 each except Htbot, 2,115) | USTC-TFC2016 (Wang et al., ICOIN 2017), processed version |

The scripts concatenate the files of a dataset, remove exact duplicate flows (17,565 in ISCX-VPN2016, 9,541 in 10 Popular Apps, 3,147 in USTC-TFC2016) and split the remaining flows with seed 2026 (Section V-A of the paper).
