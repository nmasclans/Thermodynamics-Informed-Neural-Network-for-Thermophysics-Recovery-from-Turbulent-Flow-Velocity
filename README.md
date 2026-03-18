# NN-based inference in high-pressure transcritical turbulent channel flow

This repository contains TensorFlow models and utilities used to **learn/infer thermodynamic fields** in a high-pressure transcritical turbulent channel flow. The workflow is centered around a **multi-layer perceptron (MLP)** trained with either:

- **Supervised losses** (e.g., MSE / relative errors), and/or
- A **physics-informed (PINNs-like) supervised loss** that combines a supervised regression term with **real-gas thermodynamic constraints** (Peng–Robinson Equation of State).

The main scripts are:

- `my_trainer.py`: training + validation loop, checkpoint saving
- `my_predictor.py`: load a checkpoint and generate prediction figures/metrics

The code assumes the dataset is stored as one (or more) `.npz` snapshots containing:

- `x`: the feature tensor (with a trailing dimension for feature channels)
- `features_names`: array/list of feature names
- target arrays for each target name (e.g. `c_p`, `rho`, `T`)

---

## Repository structure

- **`my_trainer.py`**: trains the MLP and validates every epoch  
- **`my_predictor.py`**: runs inference with a saved checkpoint and produces plots  
- **`my_dataset_builder.py`**: loads `.npz` snapshots, reshapes to point-wise samples, applies min–max scaling, builds `tf.data.Dataset`s  
- **`my_losses.py`**: supervised losses + PINNs-style supervised loss with real-gas constraints  
- **`my_models.py`**: Keras `Model` wrapper (`MLP`) with custom train/validate/predict loops  
- **`my_visualizers.py`**: KDE/scatter plots and mid-plane contour visualizations; classification by temperature thresholds  
- **`thermodynamics/`**: auxiliary thermodynamics functions (used by losses/analysis)  
- **`checkpoints/`**: saved weights (created/used at runtime)  
- **`figures/`**: generated plots (created/used at runtime)  
- **`results/`**: example stdout logs from training runs  

---

## Quick start

### 1) Create an environment

This project uses **TensorFlow**. Install TensorFlow in a clean environment (Conda or venv).

Example (Conda, CPU):

```bash
conda create -n pinn-rans python=3.10 -y
conda activate pinn-rans
pip install tensorflow numpy matplotlib
```

If you want GPU acceleration, install a GPU-enabled TensorFlow build that matches your CUDA setup (varies by OS/CUDA/TensorFlow version).

---

## Data format and expected keys

Training/validation/prediction use `.npz` files referenced by CLI arguments in `my_parser.py`.

The loader expects:

- **`x`**: feature array with shape similar to `(nx, ny, nz, n_features_total)`
- **`features_names`**: list/array of names for the last axis in `x`
- **Targets**: arrays for each element of `--targets_name` (default `['c_p','rho','T']`) with shape `(nx, ny, nz)`

Internally, data is reshaped to a point-wise dataset:

- features: `(nx*ny*nz, num_features_selected)`
- targets: `(nx*ny*nz, num_targets)`

### Normalization

`my_dataset_builder.py` applies **min–max scaling** using:

- `--features_limits` for each selected feature name
- `--targets_limits` for each target name

The scaled range depends on `--activation_function`:

- `relu` → \([0, 1]\)
- `tanh` → \([-1, 1]\)

---

## Training

Training script:

```bash
python my_trainer.py
```

Most configuration is provided through CLI arguments defined in `my_parser.py` (dataset paths, features/targets selection, optimizer, loss, architecture, scaling, plotting).

### Checkpoints

During training, weights are saved to `checkpoints/` according to:

- `--save_ckpt_freq` (save every N epochs; set to `0` to disable)

The model also saves an initialization checkpoint:

- `checkpoints/ckpt_initialization`

---

## Prediction / inference

Inference script:

```bash
python my_predictor.py --ckpt_filename_prediction "checkpoints/<your_run>/ckpt_E<epoch>"
```

This:

- loads the checkpoint (`--ckpt_filename_prediction`)
- runs a forward pass on the prediction dataset
- writes plots into `figures/`

Generated outputs typically include:

- KDE histograms and scatter plots per target
- mid-plane contour plots for each target (ground truth vs prediction)
- temperature-based 3-class “fluid state” plots (liquid-like / two-phase-like / gas-like), using:
  - `--T_minus` and `--T_plus`

---

## Figures and results

- Plots are written under `figures/` (created if missing).
- Example training logs are stored under `results/`.

If you run multiple experiments, consider creating subfolders under `figures/` and `checkpoints/` to keep runs separated.

---

## Notes for cluster execution

Some training runs were executed on an HPC environment; a typical pattern was:

- activate a TensorFlow GPU environment
- run with CUDA/XLA flags (system-dependent)

If you are on a cluster, adapt your job script to your site’s TensorFlow/CUDA modules and paths.

---

## Reproducibility checklist

- **Seed**: set with `--seed`
- **Scaling limits**: keep `--features_limits` and `--targets_limits` consistent across training/inference
- **Feature/target definitions**: keep `--features_idx` and `--targets_name` consistent
- **Checkpoints**: record the exact checkpoint path used for figures

---

## Citation

If you use this codebase, please cite the associated paper:

```bibtex
@article{MASCLANS2023100448,
  author  = {Masclans, N. and V{\'a}zquez-Novoa, F. and Bernades, M. and Badia, R. M. and Jofre, L.},
  title   = {Thermodynamics-informed neural network for recovering supercritical fluid thermophysical information from turbulent velocity data},
  journal = {International Journal of Thermofluids},
  volume  = {20},
  pages   = {100448},
  year    = {2023},
  issn    = {2666-2027},
  doi     = {10.1016/j.ijft.2023.100448},
  url     = {https://www.sciencedirect.com/science/article/pii/S2666202723001635}
}
```

Plain-text citation:

Masclans, N., Vázquez-Novoa, F., Bernades, M., Badia, R. M., and Jofre, L. (2023). *Thermodynamics-informed neural network for recovering supercritical fluid thermophysical information from turbulent velocity data*. **International Journal of Thermofluids**, 20, 100448. DOI: `10.1016/j.ijft.2023.100448`.
