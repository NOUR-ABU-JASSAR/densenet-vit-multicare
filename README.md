# densenet-vit-multicare

[`https://www.python.org/downloads/`](https://www.python.org/downloads/) [`https://www.tensorflow.org/`](https://www.tensorflow.org/)

## Multi-Label Thoracic Disease Classification: DenseNet‑201 vs ViT‑B/16 on MultiCaRe

This repository provides a reproducible training and evaluation pipeline comparing **DenseNet‑201** and **ViT‑B/16** for multi‑label thoracic disease classification on a curated **MultiCaRe thorax subset** (16 pathologies). The code implements patient‑level splits, Gaussian oversampling, and multi‑seed evaluation to ensure robust, leakage‑free comparisons.

**Paper**

**DenseNet-201 and ViT-B/16 for Multi-Label Thoracic Disease Classification: A Comparative Evaluation on the MultiCaRe Dataset**  
*Islam T. Almalkawi, Nour Abu Jassar, Mohammad F. Al‑Hammouri, Ali Al Bataineh, Manel Guerrero‑Zapata*

---

## Highlights

- **Label extraction:** NLP‑based extraction from case narratives with negation resolution.  
- **Data splitting:** Patient‑level splits to avoid leakage.  
- **Imbalance handling:** Gaussian noise oversampling (σ = 0.05).  
- **Evaluation:** Multi‑seed evaluation (42, 123, 256) with per‑class AUC‑ROC, accuracy, and Hamming loss.  
- **Reproducibility:** Scripts and configuration to reproduce training and evaluation runs.

---

## Key Results (3‑seed average)

| **Model**       | **Params** | **Mean AUC‑ROC** | **Hamming Loss** |
|-----------------|------------|------------------|------------------|
| DenseNet‑201    | 20.0 M     | 0.689 ± 0.004    | 0.052 ± 0.002    |
| ViT‑B/16        | 86.0 M     | 0.693 ± 0.003    | 0.049 ± 0.002    |

**Notes:** ViT‑B/16 performs better on spatially extensive conditions (cardiomegaly, pneumothorax, pleural effusion). DenseNet‑201 performs better on texture‑dependent diseases (tuberculosis, emphysema, ILD).

---

## Repository Structure

```text
.
├── Confpaper_experiments_DenseNet&ViT_for_MultiLabel_Thoracic_Disease_Classification_github.py
├── README.md
├── data/
│   └── xray_thorax_filtered/
│       ├── thorax_images/
│       └── merged_all_cleaned_scanned_per_row_labels.csv
└── paper_results/
```

- **Confpaper_experiments_...py** — Main training & evaluation script.  
- **data/xray_thorax_filtered/** — Place curated dataset here.  
- **paper_results/** — Generated CSV outputs and evaluation results.

---

## Dataset

**Source:** MultiCaRe dataset (open‑access case reports).  
**Curated subset:** MultiCaRe thorax subset.  
**Size:** 5,252 chest radiographs.  
**Labels:** 16 binary pathology labels extracted via NLP with negation resolution.  
**Patient splits:** train 2,458; validation 274; test 683.

The curated subset DOI: `10.5281/zenodo.20548927` (see citation below).

---

## Usage

1. Place the curated dataset under `data/xray_thorax_filtered/` following the structure above.  
2. Edit paths and hyperparameters in `Confpaper_experiments_DenseNet&ViT_for_MultiLabel_Thoracic_Disease_Classification_github.py`.  
3. Run the script to train and evaluate models; results will be saved to `paper_results/`.

---

## Citation

If you use this code or the curated dataset, please cite the curated dataset and the original MultiCaRe dataset:

```bibtex
@misc{jassar2026curated,
  author       = {Jassar, Nour Abu and Almalkawi, Islam and Al-Hammouri, Mohammad and Al Bataineh, Ali},
  title        = {Curated Thoracic Subset from Multi-CaRe for Multi-Label Chest X-Ray Disease Classification},
  year         = {2026},
  doi          = {10.5281/zenodo.20548927},
  url          = {https://doi.org/10.5281/zenodo.20548927}
}

@misc{offidani2025zenodo,
  author    = {M. Nievas Offidani and F. Roffet and M. C. González Galtier and M. Massiris and C. Delrieux},
  title     = {MultiCaRe Dataset},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15814064},
  url       = {https://doi.org/10.5281/zenodo.15814064}
}
```

---

## License

The MultiCaRe dataset and the curated subset are shared under **CC BY‑NC‑SA 4.0**. Check the original dataset license for additional terms.

---

## Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss the proposed change.

---

## Contact

- **Nour Abu Jassar** — 2470125@hu.edu.jo  
- **Islam Almalkawi** — eslam.malkawi@hu.edu.jo

---
