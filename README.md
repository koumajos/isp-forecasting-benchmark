# ISP forecasting benchmark

## Overview
This repository contains materials and result related to the paper "**Comparative Analysis of Deep Learning Models for Real-World ISP Network Traffic Forecasting**"
by Josef Koumar, Timotej Smoleň, Kamil Jeřábek, and Tomáš Čejka. This repository offers a benchmark for evaluating time series forecasting models 
on the CESNET24-TimeSeries dataset. 

## Repository structure
```
isp-forecasting-benchmark/
├── analysis_R2-SCORE.ipynb   # Notebook analyzing R² Score
├── analysis_RMSE.ipynb       # Notebook analyzing RMSE
├── analysis_SMAPE.ipynb      # Notebook analyzing SMAPE
├── overall_results.pdf       # Consolidated results and visualizations
├── results/                  # Directory containing model outputs and logs
├── src/                      # Source code for data processing and modeling
├── requirements.txt          # Required Python packages
├── .gitignore
└── README.md
```

## Citation
If you use this work, please cite the paper:
```
@misc{koumar2025comparativeanalysisdeeplearning,
      title={Comparative Analysis of Deep Learning Models for Real-World ISP Network Traffic Forecasting}, 
      author={Josef Koumar and Timotej Smoleň and Kamil Jeřábek and Tomáš Čejka},
      year={2025},
      eprint={2503.17410},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.17410}, 
}
```