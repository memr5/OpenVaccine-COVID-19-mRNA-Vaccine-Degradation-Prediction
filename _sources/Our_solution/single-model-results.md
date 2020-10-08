# Meet's Single Models Details

## Machine Learning Models

|Model   |Local CV   |Public LB   |Model Details   |Notebook Details   |
|---|---|---|---|---|
|XGB   |0.6597183288077114   |   |max_depth=7, SEED=2020   |[V10](https://www.kaggle.com/meemr5/openvaccine-xgb-baseline)   |
|   |   |   |   |   |
|   |   |   |   |   |

## Deep Learning Models

---

### Non-Filtered Data

|Model   |Local CV   |Public LB   |Model Details   |Notebook Details   |
|---|---|---|---|---|
|Baseline GRU   |0.38721315264701844   |0.26277   |3 Layers of GRU   |[V5](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42884202)   |
|Baseline LSTM   |0.388712215423584   |0.26521   |3 layers of LSTM   |[V6](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42893042)   |
|Multiple Parallel Layers of GRU & LSTM|0.3875190675258636|0.26319||[V7](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42899216)|
|Baseline GRU|0.38780994415283204|0.26322|New Features: (Next & Previous Loop Type)|[V9](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42926090)|
|Baseline GRU|0.38553609848022463|0.26116|New Feature: PairedWith|[V11](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42932011)|
|Baseline GRU|0.3856616675853729||Next & Prev Looptypes + PairedWith|[V12](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42937474)|
|GRU Baseline|0.38482035994529723|0.26028|PairedWith + BPPs|[V13](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42948924)|
|GRU Baseline|0.38480286598205565|0.25870|PairedWith + BPPs + BPPMaxs|[V15](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42965894)|
|GRU Baseline|0.3846119999885559|0.25833|PairedWith + BPPs + BPPMaxs + BPPMeans|[V16](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42966642)|
|GRU Baseline|0.3842911720275879|0.25717|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums|[V18](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42969547)|
|GRU Baseline|0.3844085097312927|0.25777|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPPNeighbour|[V19](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42987299)|
|GRU Baseline|0.3842735648155212|0.25783|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Prev Neighbour|[V20](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42988310)|
|GRU Baseline|0.3842824578285217|0.25668|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour|[V21](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42990156)|
|GRU(128,256,128)|0.38359965682029723|0.25891|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour|[V22](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42994498)|
|Multiple Parallel Layers of GRU & LSTM|0.38335418701171875|0.25838|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour|[V23](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=42999548)|
|GRU 4Layers|0.38397958874702454||PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums|[V24](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43008485)|
|GRU 4Layers|0.38374610543251036|0.25844|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + CyclicStep=480|[V27](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43020893)|
|GRU Baseline|0.3842721700668335|0.25683|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + Cyclic Step = 480|[V28](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43047603)|
|GRU Baseline|0.38421331644058226|0.25798|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Prev Neighbour + Cyclic Step = 480|[V29](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43051746)|
GRU Baseline|0.3836737036705017 & 0.6350993723670347|0.25779|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Prev Neighbour + BS16|[V33](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43136848)|
GRU Baseline|0.3837740898132324 & 0.6328339077104493|0.25720|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BS16|[V34](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43140011)|
GRU Baseline + Conv1D-f512-k3|0.384368896484375 & 0.6302125597507442|0.25546|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour|[V35](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43146873)|
GRU 2Layers + Conv1D-f512-k3|0.3842433154582977 & 0.6301798019513162|0.25493|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour|[V39](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43158018)|
GRU 2Layers + Conv1D-f256-k3|0.3842428088188171 & 0.6290311732639092|0.25424|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour|[V44](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43166075)|
|GRU 2Layers-128 + 2Conv1D-f256-k3|0.3841682970523834 & 0.6286693614429112|0.25602|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour|[V45](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43182145)|
1Conv1D + 2GRUs II 2LSTMs|0.3830654799938202 & 0.633038090400702 & 0.22481536017799397|0.25434|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour|[V49](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43213601)|
|Conv1D-f256-k3 + GRU 2Layers|0.3833383321762085 & 0.6295408501741827 & 0.22624839658981108|0.25416|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero|[V59](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43333635)|
|Conv1D-f256-k3 + GRU 2Layers|0.38341211080551146 & 0.6304454968436511 & 0.22621962433013812|0.25432|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Prev Neighbour + BPP_nonzero|[V60](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43336233)|
|Conv1D-f256-k3 + GRU 2Layers|0.21057788729667665 & 0.22951935575596547|0.25584|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1 + Sample_Weights|[V61](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43346922)|
|Conv1D-f256-k3 + GRU 2Layers|0.20744287371635436 & 0.22625985339822316|0.25393|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1|[V62](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43399062)|
|Transformer + 2GRUs-256|0.2097667932510376 & 0.23226156655397193|0.25835|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1|[V70](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43491219)|
3GRU-64-128-64 + Tx(dm-128,nh-4,dff-256)|0.2114489942789078 & 0.2310936371286593|0.25460|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1|[V77](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43534014)|
3GRU-64-128-64 + Tx(dm-128,nh-4,dff-256) W/O POS-E|0.2098795711994171 & 0.2303216445241878||PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1|[V83](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43591459)|
GRU-LSTM+GRU + Tx(dm-128,nh-4,dff-256) W/O POS-E|0.2088579624891281 & 0.22903614648464873||PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1|[V85](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43595128)|
|CONV + GRU + SkipCONV + 2GRU|0.20656450390815734 & 0.22462537882539144|0.25495|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1 + AbsDist between BasePair|[V89](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43614628)|

---
---

### Stratified on SNR by VP

|Model   |Local CV   |Public LB   |Model Details   |Notebook Details   |
|---|---|---|---|---|
|CONV + GRU + SkipCONV + 2GRU|0.2065820127725601 & 0.22554289782860892|0.25373|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1 + AbsDist between BasePair|[V97](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43688193)|
|CONV on OHE + CONV + GRU + SkipCONV + 2GRU|0.2065655320882797 & 0.22521942463222686 & 0.22521942463222686||PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1 + AbsDist between BasePair + OneHotEncoding|[V111](https://www.kaggle.com/meemr5/openvaccine-transformers-lstm-gru?scriptVersionId=43948406)|
|4CONV on OHE + CONV + GRU + SkipCONV + 2GRU|0.20477611422538758 & 0.2252564277098983 & 0.2252564277098983||PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1 + AbsDist between BasePair + OneHotEncoding|[V113](https://www.kaggle.com/meemr5/openvaccine-transformers-lstm-gru?scriptVersionId=43948912)|
|Conv512-3 on OHE + CONV + GRU + SkipCONV + 2GRU|0.2075606942176819 & 0.22573986884052452||PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1 + AbsDist between BasePair + OneHotEncoding|[V119](https://www.kaggle.com/meemr5/openvaccine-transformers-lstm-gru?scriptVersionId=43994018)|
|Conv128-3 on OHE + CONV + GRU + SkipCONV + 2GRU|0.20688875019550323 & 0.22619811460821349||PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1 + AbsDist between BasePair + OneHotEncoding|[V120](https://www.kaggle.com/meemr5/openvaccine-transformers-lstm-gru?scriptVersionId=43995351)|

---
---

### Filtered Data (SN_filter=1)

|Model   |Local CV   |Public LB   |Model Details   |Notebook Details   |
|---|---|---|---|---|
|GRU Baseline|Mean: 0.20845058858394622 RMSE on Scored Columns: 0.22797030496477858|0.26104|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Prev Neighbour + Cyclic Step = 480|[V30](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43063950)|

---
---

### Filtered Data (S/N>1)

|Model   |Local CV   |Public LB   |Model Details   |Notebook Details   |
|---|---|---|---|---|
|GRU Baseline|Mean: 0.21612378656864167 RMSE on Scored Columns: 0.23221140046762712|0.25755|PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Prev Neighbour + Cyclic Step = 480|[V32](https://www.kaggle.com/meemr5/openvaccine-lstm-gru?scriptVersionId=43064986)|

---
---

### Augmented Data

|Model   |Local CV   |Public LB   |Model Details   |Notebook Details   |
|---|---|---|---|---|
|CONV + GRU + SkipCONV + 2GRU|0.20700212121009826 & 0.22679129649648233||PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1 + AbsDist between BasePair + Stratified on SNR + Grouped by ids|[V126](https://www.kaggle.com/meemr5/openvaccine-transformers-lstm-gru?scriptVersionId=44073844)|
|CONV + GRU + SkipCONV + LSTM+GRU|0.2050697386264801 & 0.22628858296710197||PairedWith + BPPs + BPPMaxs + BPPMeans + BPPSums + BPP of Next Neighbour + BPP_nonzero + Val on SN_filter==1 + AbsDist between BasePair + Stratified on SNR + Grouped by ids|[Forked-V1](https://www.kaggle.com/meemr5/fork-of-openvaccine-transformers-lstm-gru?scriptVersionId=44079856)|

---
---
