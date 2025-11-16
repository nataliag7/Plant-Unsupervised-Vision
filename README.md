# Unsupervised Plant Species Identification 🌿

This repository contains code for analyzing plant images using **unsupervised machine learning** and **image quality / similarity measures**.

The goal of the project (from the original report):

> Celem niniejszej pracy jest przeprowadzenie analizy i porównania metod uczenia maszynowego bez nadzoru oraz miar oceny jakości obrazu w kontekście identyfikacji gatunków roślin. Przedstawione zostaną najważniejsze algorytmy uczenia maszynowego wykorzystane w badaniach, takie jak analiza skupień hierarchicznych (HCA) i metoda grupowania bazująca na gęstości danych (DBSCAN), a także omówione zostaną kluczowe miary oceny jakości obrazu, takie jak model opisu przestrzeni barw (HSV) oraz skalo-niezmiennicze przekształcanie cech (SIFT). Praca ta ma na celu zbadanie, w jaki sposób różne podejścia i miary mogą być zintegrowane, aby uzyskać jak najlepsze wyniki w automatycznej identyfikacji gatunków roślin.

In short: **compare different unsupervised approaches (HCA, DBSCAN) and image descriptors (HSV histograms, SIFT) for automatic plant species identification.**

---

## 📂 Project structure

```text
.
├─ src/
│  ├─ main_mse_ssim.py   # baseline: MSE + SSIM similarity to find closest species
│  ├─ dbscan_hsv.py      # DBSCAN-style pipeline using HSV color histograms
│  ├─ dbscan_sift.py     # DBSCAN-style pipeline using SIFT descriptors
│  ├─ hca_hsv.py         # hierarchical clustering (HCA) with HSV histograms
│  └─ hca_sift.py        # hierarchical clustering (HCA) with SIFT descriptors
├─ requirements.txt      # Python dependencies
├─ .gitignore
└─ README.md
