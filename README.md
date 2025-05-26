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

## 📦 Getting Started

Instructions on setting up the environment and running the code will be provided soon.

---

## 📣 Citation

If you use this code or ideas from our paper, please consider citing us. (BibTeX will be added after publication.)

---

## 📬 Contact

For any queries, please contact Pandarasamy Arjunan (samy@iisc.ac.in) or raise an issue in the repository.

---

