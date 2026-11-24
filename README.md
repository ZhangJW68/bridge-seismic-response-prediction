
<div align="center">

# Bridge Seismic Response Prediction with Machine-Learning Surrogates

**Physics-based parametric dataset generation (OpenSeesPy) + deep-learning surrogate models (CNN / LSTM / Transformer)**

<table>
  <tr>
    <td align="center"><b>Authors</b><br>Jianwei Zhang &middot; Chao Yu &middot; Xiya Liu</td>
    <td align="center"><b>Affiliation</b><br>College of Civil Engineering, Tongji University</td>
    <td align="center"><b>License</b><br>Copyright &copy; 2025 &middot; All rights reserved</td>
  </tr>
</table>

<p>
  <img alt="python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="openseespy" src="https://img.shields.io/badge/OpenSeespy-Finite--Element-orange">
  <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-Surrogate%20Models-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="pydantic" src="https://img.shields.io/badge/Pydantic-v2-293237">
  <img alt="license" src="https://img.shields.io/badge/License-Proprietary-lightgrey">
</p>

</div>

---

## Overview

This repository provides an end-to-end framework for **rapid seismic response prediction of bridge piers**. Conventional nonlinear time-history analysis with OpenSeesPy is accurate but computationally expensive, which makes large-scale parametric studies, fragility assessment, and design exploration impractical. Here we bridge the gap with a two-stage pipeline:

1. **Dataset generation** &mdash; Latin Hypercube Sampling (LHS) drives an OpenSeesPy parametric seismic analysis over a bridge/pier design space, producing input-parameter &rarr; engineering-demand-parameter (EDP) pairs from nonlinear time-history analyses against recorded ground motions.
2. **Surrogate modeling** &mdash; three deep-learning surrogates (**CNN**, **LSTM**, **Transformer**) learn the mapping from structural parameters + ground-motion features to peak seismic demand, enabling millisecond-level prediction in place of seconds-to-minutes per OpenSeesPy run.

The framework is engineered for reproducibility and portability: every path is **relative to the script directory** and overridable from the command line, with no absolute or machine-specific addresses hard-coded.

---

## Key Features

| Capability | Detail |
| --- | --- |
| **Validated parameter model** | Pydantic `BaseModel` enforces physical bounds on every parameter; cross-section properties (area *A*, inertia *Iz*) are auto-computed via `@computed_field`. |
| **Latin Hypercube Sampling** | Deterministic, seed-controlled uniform LHS gives space-filling coverage of the design space. |
| **OpenSeesPy nonlinear time history** | Modeling &rarr; gravity static analysis &rarr; seismic transient analysis (Newmark 0.5/0.25, uniform excitation) with per-step EDP extraction. |
| **Two structural archetypes** | *Cantilever pier* (1 m `dispBeamColumn` + circular fiber section, P-Delta) and *three-span continuous bridge* (elastic beam-column piers + deck). |
| **Multiprocessing + live progress** | Sample-level parallel workers feed a progress queue; results aggregate to a tidy CSV. |
| **Three deep surrogates** | CNN / LSTM / Transformer share one dataset; the Transformer fuses a 1-D CNN ground-motion encoder with a bridge-parameter MLP through a Transformer encoder. |
| **Portable & GitHub-ready** | `argparse` CLI, relative defaults (`./waves`, `./time_history_results`), English-only source, copyright headers. |

---

## Architecture

```text
        design space                     OpenSeesPy                      surrogate models
  +----------------------+         +-------------------+         +--------------------------+
  |  LHS parameter       |  N x M  |  nonlinear time   | labels  |  CNN   --+               |
  |  sampling            +-------->+  history analysis +-------->+  LSTM  --+-- peak demand |
  |  (h, d, m / span...) | samples |  per ground motion|  (EDP)  |  Transp  --+  prediction |
  +----------------------+         +-------------------+         +--------------------------+
       paramters_sampling          strucrure_modeling /              Transformer / CNN / LSTM
       (cantilever_demo)            data_generation
```

**Data flow.** LHS produces *N* parameter samples &rarr; each is analyzed against *M* ground motions by OpenSeesPy &rarr; peak top/mid displacement (the EDP) is recorded into a CSV &rarr; the CSV is split, standardized, and used to train each surrogate &rarr; model accuracy is reported by R&sup2;, RMSE, MAE with prediction-vs-truth and residual plots.

---

## Repository Structure

```text
.
|-- cantilever_demo.py            # Entry point: cantilever-pier LHS study + multiprocessing scheduler
|-- paramters_sampling.py         # PierParams (Pydantic) + LHSSampler + generate_pier_samples()
|-- strucrure_modeling.py         # run_one_earthquake(): modeling -> gravity -> seismic time history
|-- data_generation.py            # Three-span continuous bridge parametric dataset generator (CLI)
|-- cnn_model.py                        # CNN surrogate: train / evaluate on the generated dataset
|-- lstm_model.py                       # LSTM surrogate: sequence model of seismic demand
|-- transformer_model.py                # Enhanced Transformer surrogate (CNN acc-encoder + Transformer)
|-- utils/                        # Record.AccRecord (gm reader) + progress helpers
`-- waves/                        # Ground-motion acceleration records: 1.acc, 2.acc, ... (dt = 0.02 s)
```

> A shared utility package `utils` provides `Record.AccRecord` (ground-motion file reader) and `progress.progress_map` / `send_progress_message` (parallel progress reporting).

---

## Structural Models & Parameter Space

### Cantilever Pier (`paramters_sampling.py`)

| Parameter | Symbol | Range | Description |
| --- | --- | --- | --- |
| Pier height | `h` | 8 &ndash; 20 m | Free height of the cantilever pier |
| Pier diameter | `d` | 1.5 &ndash; 3.0 m | Circular cross-section diameter |
| Top mass | `m_top` | 200 &ndash; 1000 t | Superstructure concentrated mass (kN&middot;s&sup2;/m in OpenSees) |
| Section area | `A` | computed | `pi * d^2 / 4` |
| Inertia | `Iz` | computed | `pi * d^4 / 64` |

### Three-Span Continuous Bridge (`data_generation.py`)

| Parameter | Symbol | Range |
| --- | --- | --- |
| Span length | `span_length` | 30 &ndash; 50 m |
| Pier height | `pier_height` | 10 &ndash; 20 m |
| Pier diameter | `pier_diameter` | 1.5 &ndash; 3.0 m |
| Deck mass / unit length | `deck_mass` | 8 &ndash; 15 t/m |

Both models use concrete modulus `E = 30 GPa`, fixed pier bases, and the same gravity &rarr; transient analysis sequence.

---

## Surrogate Models

All surrogates consume the same parametric dataset and share a common feature layout: a small vector of structural parameters concatenated with ground-motion information.

- **CNN** &mdash; learns local patterns over the input feature/sequence via stacked convolutions; a lightweight and robust baseline.
- **LSTM** &mdash; captures temporal dependencies in the ground-motion / demand sequence, well suited to time-history-style inputs.
- **Transformer (`EnhancedTransformerModel`)** &mdash; a hybrid encoder:
  - a **1-D CNN** extracts features from the acceleration time history,
  - an **MLP** encodes the structural parameters,
  - both branches are fused and passed through a multi-head **Transformer encoder** (8 heads, 6 layers, `d_model = 256`),
  - regression heads output the scalar peak demand; hand-crafted ground-motion features (RMS, dominant frequency, energy distribution across quartiles, peak/duration statistics) augment the raw samples.

**Evaluation.** Test-set metrics: MSE, RMSE, MAE, R&sup2;; visualization via loss curves, predicted-vs-true scatter, and residual plots.

---

## Installation

```bash
# Python >= 3.10
git clone <your-repo-url>
cd <repo>
pip install numpy pandas scipy matplotlib scikit-learn \
            openseespy pydantic torch
```

A CUDA-capable GPU is used automatically when available; otherwise training falls back to CPU.

---

## Usage

### 1. Generate the dataset &mdash; cantilever pier

Place acceleration records as `waves/1.acc`, `waves/2.acc`, ... (`dt = 0.02 s`), then:

```bash
python cantilever_demo.py
```

The script performs LHS over the pier parameter space, runs every sample through all ground motions in parallel, and writes `pier_lhs_results.csv` (columns: `sample_id, gm_id, h_m, d_m, m_top_t, max_disp_top_m`).

### 2. Generate the dataset &mdash; three-span bridge

```bash
python data_generation.py \
    --gm-folder ./waves \
    --plot-dir  ./time_history_results
```

All paths default to the script directory and can be overridden (`--gm-folder`, `--gm-start`, `--gm-count`, `--n-samples`, `--result-csv`, `--plot-dir`, `--seed`). Per-sample CSVs and displacement time-history plots are written under `time_history_results/sample_XXX/`.

### 3. Train & evaluate a surrogate

```bash
python Transformer.py --results-dir ./time_history_results --acc-dir ./waves
# equivalently
python CNN.py
python LSTM.py
```

The model loads the generated dataset, builds train/validation/test splits (60 / 20 / 20), standardizes features, trains with `AdamW` + `ReduceLROnPlateau`, and reports test metrics with diagnostic plots.

---

## Output Format

The dataset is a tidy CSV &mdash; one row per (sample x ground motion):

| column | meaning |
| --- | --- |
| `sample_id` / `gm_id` | parameter sample & ground-motion identifiers |
| `h_m`, `d_m`, `m_top_t` *(pier)* | structural parameters |
| `span_length_m`, `pier_height_m`, `pier_diameter_m`, `deck_mass_t_per_m` *(bridge)* | structural parameters |
| `max_disp_top_m` / `max_disp_mid_m` | peak displacement demand (EDP, the regression label) |

---

## Tech Stack

`Python` &middot; `OpenSeesPy` (FEM, nonlinear time history) &middot; `Pydantic` (validated parameter models) &middot; `NumPy` / `SciPy` (LHS, FFT, peak stats) &middot; `scikit-learn` (split / scale / metrics) &middot; `PyTorch` (surrogate networks) &middot; `multiprocessing` &middot; `matplotlib`

---

## Design Notes

- **No hard-coded absolute paths** &mdash; every file/folder is resolved from `Path(__file__).resolve().parent`, so the project runs identically after `git clone` on any machine.
- **Exception-safe cleanup** &mdash; `finally` blocks perform `ops.wipe()` only; `return` statements live outside `finally` so errors are never silently swallowed.
- **Reproducible sampling** &mdash; fixed LHS seeds yield an identical design-of-experiments across reruns.

---

## Citation & License

If you build upon this work, please cite:

> ZHANG Jianwei, YU Chao, LIU Xiya. Bridge Seismic Response Prediction with Machine-Learning Surrogates[CP/OL]. College of Civil Engineering, Tongji University, 2025. (https://github.com/ZhangJW68/bridge-seismic-response-prediction).

Copyright &copy; 2025 Jianwei Zhang, Chao Yu, Xiya Liu. All rights reserved.
