# Stock Network Community Detection and Node Migration Dataset

This repository contains the dataset accompanying a research paper. It includes **stock market data**, **partial-correlation networks**, **community detection results**, **node migration matrices/paths**, and **panel regression data**, with time windows from 2014-01 to 2016-11 (35 windows in total).

---

## Directory Structure

| Directory | Description |
|-----------|-------------|
| **1_Stocks** | Stock data: close prices and log returns; constituents are SSE 180 (SZ180) |
| **2_Networks** | Stock networks built from partial correlation of log returns: partial correlation matrices (xlsx), P-threshold networks (csv/gml) |
| **3_Communities** | Community assignments, modularity Q values, community evolution figures (png), and VGAER community detection code |
| **4_Node migration matrix** | Node migration matrices (top/bottom) and enrichment analysis results |
| **5_Node migration path** | Node migration paths and industry classification (JSON) |
| **6_Panel Data Regression** | Panel regression data: community yields, attractiveness, merged community data, etc. (JSON) |

---

## 1_Stocks

- **SZ180_stocks/** — Per-stock data for each constituent (xlsx, named by ticker e.g. `600000.XSHG.xlsx`)
- **Stocks_close price/** — Windowed close prices for the full universe
- **Stocks_close price_log returns/** — Windowed log returns for the full universe

---

## 2_Networks

- **Partial corr matrix/** — Partial correlation matrices per time window (xlsx), used to build networks
- **P Threshold value networks/** — Network edge lists (csv) and graph files (gml) at chosen P thresholds

---

## 3_Communities

- **community assignments/** — Community membership per window (xlsx)
- **community_changes.xlsx** — Community change records
- **q_values.xlsx** — Modularity Q values
- **\*.png** — Community structure visualizations per window
- **VGAER community detection algorithm code/** — Python implementation of the VGAER (probability-based variational graph) community detection algorithm

---

## 4_Node migration matrix

- **Node migration matrix_top.xlsx / _bottom.xlsx** — Node migration matrices (top/bottom split)
- **enrichment_*.xlsx** — Enrichment analysis for migrations (all / top50 / bottom50)

---

## 5_Node migration path

- **migration_paths.json** — Node migration paths
- **industry_classification.json** — Industry classification

---

## 6_Panel Data Regression

- **panel_data.json** — Raw/processed panel data for regression
- **community_yield_*.json** — Community yields, beta, standard deviation, turnover rate, etc.
- **community_attractiveness.json** — Community attractiveness
- **merged_community_data*.json** — Merged community data (with/without attractiveness)

---

## Usage

1. **Reproduce network construction**: Compute partial correlations from log returns in `1_Stocks` to obtain matrices and networks in `2_Networks`.
2. **Reproduce community detection**: Use the code and instructions in `3_Communities/VGAER community detection algorithm code/` (see README and Instruction.docx there).
3. **Panel regression**: Use the JSON files in `6_Panel Data Regression` directly for regression or visualization.

---

## Data Formats

- **Tables**: xlsx (Excel)
- **Graphs**: csv (edge list), gml (graph)
- **Structured data**: JSON

---

## Citation and License

If you use this dataset or the VGAER code in your research, please cite the data source and the associated paper (publication details can be added here once available).

---

*Last updated: 2025-02*
