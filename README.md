# Motion Diffusion Model – Ablation Study on Diffusion Transformers

This repository contains an **ablation study** on the [Motion Diffusion Model (MDM)](https://github.com/GuyTevet/motion-diffusion-model/tree/main),  
exploring the integration of **Diffusion Transformer (DiT)** components, attention mechanisms, and tokenization strategies.

---

## 🔍 Overview

Our project systematically investigates how architectural modifications inspired by **Diffusion Transformers (DiT)** affect motion generation quality within the MDM framework.  
Specifically, we analyze:

- The effect of **AdaLN-Zero conditioning** on model stability and convergence.  
- The influence of **temporal Sliding-Window attention** and **Neighborhood attention** mechanisms.  
- The impact of different **temporal tokenization methods**, including **frame**, **temporal patch**, and **conv-temporal** tokens.

All quantitative and qualitative results are included in the [`results`](./results) directory,  
and summarized in the comprehensive report [`Results.pdf`](./results/Results.pdf).

---

## ⚙️ Setup & Reproducibility

To reproduce our experiments:

1. **Clone and install the original MDM repository** following the setup instructions here:  
   👉 [https://github.com/GuyTevet/motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model/tree/main)

2. **Replace the original file** `model/mdm.py`  
   with one of our modified versions from the [`ableation_code`](./ableation_code) directory —  
   each file corresponds to a different ablation configuration (attention or tokenization variant).

3. **Run training and evaluation** using the provided helper scripts:  
   - 🧩 `train_on_server.sh` – convenient training script for server execution.  
   - 📈 `eval_script_on_server.sh` – evaluation script to reproduce quantitative results.

All resulting logs, checkpoints, and metrics will be stored under the `results` directory.

---

## 📊 Results Summary

All individual experiment outputs (FID, R-Precision, Diversity, Multimodality) are available in [`results/`](./results).  
For convenience, the complete summary of all experiments is compiled in  
[`Results.pdf`](./results/Results.pdf), including tables and figures used in the final report.

---

## 📁 Repository Structure

