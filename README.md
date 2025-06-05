# Mix-BEATS: Mixer-enhanced Basis Expansion Analysis for Load Forecasting

This repository contains the official code for the paper:

**Mix-BEATS: Mixer-enhanced Basis Expansion Analysis for Load Forecasting**  
*Anuj Kumar, Harish Kumar Saravanan, Shivam Dwivedi, Pandarasamy Arjunan*  
*Robert Bosch Centre for Cyber Physical Systems, Indian Institute of Science, Bangalore*  
To appear in **ACM e-Energy 2025**, June 17, 2025, Rotterdam, Netherlands.

---
## 📄 Paper

[Mix-BEATS: Mixer-enhanced Basis Expansion Analysis for Load Forecasting (E-Energy ’25)](http://camps.aptaracorp.com/ACM_PMS/PMS/ACM/EENERGY25/49/65dd3077-2f5b-11f0-ada9-16bb50361d1f/OUT/eenergy25-49.html)

---

## 🧠 Overview

Mix-BEATS is a lightweight, hybrid model for short-term load forecasting that combines the residual learning of N-BEATS with the MLP-based patch and time mixing of TSMixer. Designed for efficiency and generalization, it achieves superior performance across diverse buildings while being suitable for edge deployment.

---

## Features

- **Hybrid Architecture Combining N-BEATS and TSMixer**  
  Leverages residual learning with patch and time mixing for efficient and accurate forecasting.

- **Mixer-based Temporal Modeling**  
  Applies MLP-based patch and time mixing operations inspired by vision transformers for effective time-series representation.

- **Basis Expansion Analysis**  
  Uses N-BEATS-style basis functions to improve interpretability and feature extraction.

- **Pretrained on Large-Scale, Real-World Smart Meter Data**  
  Trained on hourly consumption data from over 38,000 buildings to ensure robust generalization.

- **Lightweight and Edge-Deployable**  
  Optimized for computational efficiency, making it suitable for deployment in resource-constrained environments.

- **Comprehensive Benchmarking**  
  Evaluated against state-of-the-art time series foundation models and generic models in zero-shot, fine-tuned, and domain-specific scenarios.

- **Open and Reproducible**  
  Publicly available codebase with training, evaluation, and benchmarking scripts for easy replication and extension.

---
## 📊 Real-World Building Datasets

This project uses large-scale **real-world building energy datasets** from commercial and residential domains, collected from multiple countries.

| Dataset   | Location     | Type        | # Buildings | # Observations | Years       |
|-----------|--------------|-------------|-------------|----------------|-------------|
| IBlend    | India        | Commercial  | 9           | 296,357        | 2013–2017   |
| Enernoc   | USA          | Commercial  | 100         | 877,728        | 2012        |
| NEST      | Switzerland  | Residential | 1           | 34,715         | 2019–2023   |
| Ireland   | Ireland      | Residential | 20          | 174,398        | 2020        |
| MFRED     | USA          | Residential | 26          | 227,622        | 2019        |
| CEEW      | India        | Residential | 84          | 923,897        | 2019–2021   |
| SMART*    | USA          | Residential | 114         | 958,998        | 2016        |
| Prayas    | India        | Residential | 116         | 1,536,409      | 2018–2020   |
| NEEA      | USA          | Residential | 192         | 2,922,289      | 2018–2020   |
| SGSC      | Australia    | Residential | 13,735      | 172,277,213    | 2011–2014   |
| GoiEner   | Spain        | Residential | 25,559      | 632,313,933    | 2014–2022   |

**Total: 39,956 buildings and 812M+ hourly observations**

> ⚠️ These datasets are used under their respective terms/licenses for academic research only.

---

## 📚 Generic Benchmark Datasets

TSPulse is extensively evaluated on popular time-series forecasting benchmarks from domains such as energy, economics, traffic, weather, and disease surveillance.

| Dataset         | Variates | Timesteps | Granularity |
|------------------|----------|-----------|-------------|
| Weather          | 21       | 52,696    | 10 min      |
| Traffic          | 862      | 17,544    | 1 hour      |
| Electricity      | 321      | 26,304    | 1 hour      |
| ETTh1            | 7        | 17,420    | 1 hour      |
| ETTh2            | 7        | 17,420    | 1 hour      |
| ETTm1            | 7        | 69,680    | 15 min      |
| ETTm2            | 7        | 69,680    | 15 min      |
| Illness (ILI)    | 7        | 966       | 1 week      |
| Exchange-Rate    | 8        | 7,588     | 1 day       |

> 📌 Train/Val/Test Splits:
> - ETT: 60% / 20% / 20%
> - Others: 70% / 10% / 20%



---

## 📦 Getting Started

Instructions on setting up the environment and running the code will be provided soon.

---

## 📣 Citation

If you use this code or ideas from our paper, please consider citing us. (BibTeX will be added after publication.)

---

## 📬 Contact

For any queries, please contact Pandarasamy Arjunan (samy@iisc.ac.in) or raise an issue in the repository.

---

