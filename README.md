# NeuroAI: Metaheuristic Optimization-Enhanced Deep Learning Framework for Brain Tumor MRI Detection & Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end medical AI research and clinical diagnosis framework that integrates **Nature-Inspired Swarm Intelligence & Evolutionary Metaheuristic Optimizers** (GWO, PSO, WOA, GA, HHO, ChOA) with **Deep Convolutional Neural Networks & Vision Attention Modules** (EfficientNet, ResNet, DenseNet, CBAM Attention CNN) for multi-class brain tumor detection, deep feature selection, multi-model ensemble weight blending, hyperparameter tuning, Explainable AI (Grad-CAM / Grad-CAM++), and automated clinical PDF diagnostic report generation.

---

## Key Highlights

- **4-Class Brain MRI Classification**: Highly accurate detection across `Glioma`, `Meningioma`, `Pituitary Tumor`, and `Healthy / No Tumor`.
- **6 Nature-Inspired Metaheuristic Optimizers**:
  1. **Grey Wolf Optimizer (GWO & Binary GWO)**: Leadership hunting hierarchy ($\alpha, \beta, \delta$).
  2. **Particle Swarm Optimization (PSO & Binary PSO)**: Swarm velocity and personal/global bests ($p_{best}, g_{best}$).
  3. **Whale Optimization Algorithm (WOA & Binary WOA)**: Bubble-net spiral hunting mechanisms.
  4. **Harris Hawks Optimization (HHO)**: Dynamic transition between soft/hard besieges and Lévy flights.
  5. **Genetic Algorithm (GA & Binary GA)**: Tournament selection, two-point crossover, bit-flip mutation, and elitism.
  6. **Chimp Optimization Algorithm (ChOA)**: Specialized group hunting roles (Drivers, Chasers, Blockers, Attackers).
- **3 Optimization Dimensions**:
  - **Level 1: Binary Deep Feature Selection**: Eliminates 40%–60% of redundant CNN embedding dimensions while improving classification accuracy.
  - **Level 2: Multi-Model Ensemble Blending**: Searches for continuous simplex blending weights $\mathbf{w} \in \Delta^K$ across EfficientNet, ResNet, DenseNet, and CBAM Attention CNN.
  - **Level 3: Hyperparameter Tuning (HPO)**: Optimizes learning rate, weight decay, and dropout rate.
- **Medical MRI Preprocessing Pipeline**:
  - Skull stripping via extreme point contour detection.
  - Contrast Limited Adaptive Histogram Equalization (CLAHE).
  - Edge-preserving bilateral denoising.
- **Explainable AI (XAI) & Lesion Localization**:
  - Grad-CAM & Grad-CAM++ saliency heatmaps with interactive opacity slider.
  - Automated tumor bounding box extraction, center of mass, and estimated tumor burden percentage.
- **Full-Stack Clinical Web Platform**:
  - Real-time diagnostic viewer, live Chart.js convergence curves, model benchmarking suite, patient database, and automated clinical PDF report generation (via ReportLab).

---

## System Architecture

```text
Brain MRI Scan (T1-CE / Axial)
      │
      ▼
┌───────────────────────────────────────────────────────────┐
│ Medical Preprocessing Pipeline                             │
│ • Extreme Point Contour Skull Stripping                   │
│ • CLAHE Contrast Enhancement & Bilateral Denoising        │
│ • Normalization & Data Augmentation                       │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ Deep Vision Feature Extractors                             │
│ • AttentionBrainCNN (Dual-Branch CBAM Attention)          │
│ • EfficientNet-B0 / B3                                    │
│ • DenseNet-121                                            │
│ • ResNet-50                                               │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ Metaheuristic Optimization Engine                          │
│ [GWO | PSO | WOA | GA | HHO | ChOA]                       │
│ • Binary Feature Selection (Sparse Discriminative Subspace)│
│ • Continuous Multi-Model Ensemble Blending (Simplex w_i)  │
│ • Continuous Hyperparameter Optimization (LR, WD, Dropout) │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ Clinical Decision Support & Explainable AI (XAI)          │
│ • 4-Class Diagnostic Verdict & Confidence Breakdown       │
│ • Grad-CAM / Grad-CAM++ Saliency Heatmap Overlay          │
│ • Tumor Bounding Box, Quadrant & Area Burden (%)         │
│ • Downloadable Clinical PDF Diagnostic Report             │
└───────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```text
brainDetection/
├── app/
│   ├── static/
│   │   ├── css/style.css         # Modern Glassmorphic Dark Design System
│   │   └── js/app.js             # Frontend logic, Chart.js, Grad-CAM slider
│   ├── templates/
│   │   └── index.html            # Clinical Dashboard Single-Page Interface
│   ├── database.py               # SQLAlchemy SQLite patient & experiment models
│   └── server.py                 # Flask REST API endpoints
├── core/
│   ├── metaheuristics/           # Swarm intelligence & evolutionary optimizers
│   │   ├── base.py               # Base class & convergence trackers
│   │   ├── gwo.py                # Grey Wolf Optimizer (GWO & BGWO)
│   │   ├── pso.py                # Particle Swarm Optimization (PSO & BPSO)
│   │   ├── woa.py                # Whale Optimization Algorithm (WOA & BWOA)
│   │   ├── ga.py                 # Genetic Algorithm (GA & BGA)
│   │   ├── hho.py                # Harris Hawks Optimization (HHO)
│   │   ├── choa.py               # Chimp Optimization Algorithm (ChOA)
│   │   ├── feature_selector.py   # Metaheuristic Deep Feature Selection
│   │   ├── ensemble_optimizer.py # Metaheuristic Ensemble Weight Blending
│   │   └── hyperparameter_tuner.py # Metaheuristic HPO
│   ├── dataset.py                # Kaggle dataset loader & synthetic MRI generator
│   ├── models.py                 # EfficientNet, ResNet, DenseNet, CBAM Attention CNN
│   ├── preprocessing.py          # Skull stripping, CLAHE, denoising
│   ├── report_generator.py       # Clinical PDF Report Generator (ReportLab)
│   ├── trainer.py                # PyTorch training loop & cross-validation metrics
│   └── xai.py                    # Grad-CAM, Grad-CAM++, tumor bounding box
├── data/
│   └── sample_mris/              # Benchmark MRI samples across all 4 classes
├── experiments/
│   ├── run_optimization.py       # CLI metaheuristic comparative benchmark
│   └── train_model.py            # CLI deep learning training script
├── tests/                        # Comprehensive automated unit test suite
├── run.py                        # Single-command application launch entrypoint
└── requirements.txt              # Dependency specifications
```

---

## Installation & Setup

1. **Clone or Navigate to the Workspace**:
   ```powershell
   cd c:\Users\rraje\OneDrive\Documents\brainDetection
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## Running the Web Platform

Launch the full interactive clinical platform with a single command:

```powershell
python run.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

### Web Platform Features:
1. **Clinical AI Diagnosis**: Drag-and-drop any MRI scan or click a sample (Glioma, Meningioma, Pituitary, Normal). Adjust preprocessing parameters, view real-time predictions, inspect the Grad-CAM heatmap with the interactive opacity slider, and download a publication-ready clinical PDF report.
2. **Metaheuristic Lab**: Select any of the 6 optimizers (GWO, PSO, WOA, GA, HHO, ChOA) and run live feature selection or ensemble blending with real-time Chart.js convergence curves.
3. **Model Benchmarks**: View multi-model metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Latency) and interactive radar/bar comparisons.
4. **Patient Database**: Search past scan records, view severity triage indicators, and re-download diagnostic reports.

---

## Running Standalone CLI Experiments

### 1. Benchmark All 6 Metaheuristic Algorithms:
```powershell
python experiments/run_optimization.py --population 20 --iterations 25
```

### 2. Train Deep Learning Backbone on Kaggle MRI Dataset:
```powershell
python experiments/train_model.py --model attention_cnn --epochs 10 --batch_size 32
```

---

## Running Automated Tests

Run the complete unit and integration test suite:
```powershell
python tests/test_preprocessing.py
python tests/test_metaheuristics.py
python tests/test_models.py
python tests/test_report.py
```
*(All 13 unit test cases pass with 100% verification).*

---

## Citations & References

- **GWO**: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). *Grey Wolf Optimizer*. Advances in Engineering Software.
- **PSO**: Kennedy, J., & Eberhart, R. (1995). *Particle Swarm Optimization*. IEEE ICNN.
- **WOA**: Mirjalili, S., & Lewis, A. (2016). *The Whale Optimization Algorithm*. Advances in Engineering Software.
- **HHO**: Heidari, A. A., et al. (2019). *Harris Hawks Optimization: Algorithm and Applications*. FGCS.
- **ChOA**: Khishe, M., & Mosavi, M. R. (2020). *Chimp Optimization Algorithm*. Expert Systems with Applications.
- **Grad-CAM**: Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV.
