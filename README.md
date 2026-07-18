# Off-shell angular-projection study

This repository contains a reproducible analysis prototype for extracting differential angular coefficients from POWHEG LHE events in

$$H^{(*)}\to Z_1Z_2\to \mu^+\mu^-e^+e^-.$$

The first notebook reads a local LHE file, removes the POWHEG recoil with a Born-like Lorentz map, constructs the four-lepton angles, and trains a conditional angular-moment estimator using the density-ratio machinery in [`nsbi-common-utils`](https://github.com/iris-hep/nsbi-lhc-toolkit).

## Analysis contents

```text
├── notebooks/
│   └── 01_lhe_angular_coefficients.ipynb
├── src/offshell_angles/
│   ├── harmonics.py    # symmetric spherical harmonics, bounds, BCE targets
│   ├── kinematics.py   # Born projection and angle definitions
│   ├── lhe.py          # pylhe object selection and pandas records
│   └── training.py     # duplicated weights and ratio normalization correction
├── tests/              # synthetic kinematics and a minimal LHE fixture
├── vendor/             # pinned pure-Python nsbi-common-utils wheel and provenance
├── pixi.toml           # complete analysis environment
└── pyproject.toml
```

Generated models, plots, local LHE files, and Pixi environments are intentionally ignored by Git.

## Physics conventions

The implementation makes the following choices explicit:

- `Z1` is always the dimuon system and `Z2` is always the dielectron system. No mass ordering is used in the off-shell region.
- Only the final-state $e^-$, $e^+$, $\mu^-$, and $\mu^+$ records are read. The two $Z$ candidates and the Higgs candidate are reconstructed from their four-vector sums; LHE IDs 23 and 25 are deliberately ignored.
- The harmonic coordinates $\Omega_i=(\theta_i,\phi_i)$ use the positively charged lepton in its parent-$Z$ rest frame.
- In the Born-projected four-lepton rest frame, $\hat z_i$ follows $Z_i$, $\hat y_i$ is normal to the beam--$Z_i$ production plane, and $\hat x_i=\hat y_i\times\hat z_i$.
- The separately named `theta1_standard` and `theta2_standard` follow the negatively charged-lepton convention of [arXiv:1208.4018](https://arxiv.org/abs/1208.4018).
- The expansion includes the requested normalization,

  $$
  p(\Omega_1,\Omega_2,x)
  =\frac{1}{4\pi}\sum_{\alpha\preceq\beta}
  \mathcal S_{\alpha\beta}(x)\,
  \mathcal Y^{(+)}_{\alpha\beta}(\Omega_1,\Omega_2),
  $$

  so that

  $$
  \mathcal S_{\alpha\beta}(x)
  =4\pi\int d\Omega_1d\Omega_2\,
  \mathcal Y^{(+)*}_{\alpha\beta}(\Omega_1,\Omega_2)
  p(\Omega_1,\Omega_2,x).
  $$

  Since $\mathcal Y^{(+)}_{00;00}=1/(4\pi)$, angular integration returns $\mathcal S_{00;00}$.

The recoil removal follows Eqs. (2.14)--(2.18) of [arXiv:2606.11083](https://arxiv.org/abs/2606.11083). For the four-lepton momentum $k$, the code applies $B_L$, then $B_T$, then $B_L^{-1}$ to every lepton. This preserves $m_{4\ell}$, $y_{4\ell}$, and all internal invariant masses while setting $p_{T,4\ell}$ to zero. Every event is checked numerically.

## Run at the UChicago Analysis Facility with Pixi

The instructions below assume the LHE file is already on storage visible from the notebook pod.

### 1. Start JupyterLab

1. Sign in at [af.uchicago.edu](https://af.uchicago.edu/).
2. Open **Services → JupyterLab**.
3. Give the notebook server a short name without spaces.
4. Choose the resources for the intended execution mode:
   - **CPU:** request a few CPU cores, sufficient memory for the LHE sample, and **0 GPU instances**.
   - **GPU:** request at least **1 GPU instance**, choose an available GPU-memory option, and allocate enough CPU memory for LHE parsing and preprocessing. A Pixi environment cannot expose a GPU that was not assigned to the JupyterLab pod.
5. Select `ml_platform:latest`, the facility's recommended ML image with CUDA support, and launch the server.

The current resource limits and launch options are documented by the [UChicago Analysis Facility](https://usatlas.github.io/af-docs/uchicago/jupyter/). They can change, so use that page as the authoritative source.

### 2. Clone the private repository

Open a JupyterLab terminal and clone with either an SSH key registered at GitHub or an authenticated HTTPS session:

```bash
git clone git@github.com:rafaellopesdesa/offshell_angular_coefficients.git
cd offshell_angular_coefficients
```

For HTTPS, use the repository's GitHub clone URL and the facility-supported credential flow. Do not put a personal access token in a notebook or committed file.

### 3. Create the CPU or GPU project environment

Pixi is provided by the `ml_platform` image. The project deliberately defines two reproducible environments:

| Environment | PyTorch variant | Use it when |
|---|---|---|
| `analysis` | `pytorch-cpu` | The JupyterLab pod has no GPU, or only preprocessing and validation are needed. |
| `analysis-gpu` | `pytorch-gpu` with a CUDA 13 system requirement | The JupyterLab pod was launched with one or more GPU instances. |

For CPU execution, run from the repository root:

```bash
pixi install -e analysis
pixi run -e analysis test
```

For GPU execution, run:

```bash
pixi install -e analysis-gpu
pixi run -e analysis-gpu test
pixi run -e analysis-gpu python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

The final command should print a non-`None` CUDA version, `True`, and the assigned NVIDIA device name. The tests exercise the analysis code but do not by themselves prove that the GPU is visible.

Both environments include `pylhe`, `vector`, `mplhep`, SciPy, pandas, PyTorch/Lightning, ONNX Runtime, Jupyter support, and a commit-pinned build of `nsbi-common-utils`. The CPU environment avoids downloading CUDA libraries, while the GPU environment constrains conda-forge to select a CUDA-enabled PyTorch build.

The toolkit wheel is stored under `vendor/` and was built from commit `fc09848fc6540fd32310faebbe9db6eea7ecd17b` of `iris-hep/nsbi-lhc-toolkit`. This avoids checking out the upstream repository during the Pixi solve: that repository contains Git LFS rules for large example files, while the analysis needs only its small pure-Python package. The wheel's provenance, rebuild command, and SHA-256 digest are recorded in `vendor/README.md`.

The committed `pixi.lock` may initially predate the GPU feature. The first `pixi install -e analysis-gpu` refreshes an out-of-date lock unless `--locked` is supplied. After validating the GPU solve at the AF, commit the refreshed lock file so later sessions reconstruct the exact CPU and GPU package builds.

### 4. Select the Pixi kernel

Open `notebooks/01_lhe_angular_coefficients.ipynb`, then follow the facility's `pixi-kernel` workflow:

1. Click the kernel selector in the upper-right corner and select **pixi**.
2. Open the property inspector using the gear icon in the right sidebar.
3. Select `analysis` for CPU execution or `analysis-gpu` for GPU execution.
4. Save the notebook.
5. Restart the kernel.

The saved error output in the training cell came from `.pixi/envs/default`, which had a CPU-only PyTorch build. Selecting the generic **pixi** kernel is not sufficient by itself: the environment in the property inspector must match the requested execution mode.

If the desired environment is not listed, confirm that the notebook was opened from the cloned project and run one of:

```bash
# CPU
pixi install -e analysis
pixi list -e analysis pixi-kernel

# GPU
pixi install -e analysis-gpu
pixi list -e analysis-gpu pixi-kernel
```

Then restart JupyterLab and repeat the selection. The [facility Pixi instructions](https://usatlas.github.io/af-docs/uchicago/jupyter/#pixi) contain the current UI workflow and troubleshooting advice.

As a terminal-only fallback, register a mode-specific conventional kernel:

```bash
# CPU
pixi run -e analysis kernel-cpu

# GPU
pixi run -e analysis-gpu kernel-gpu
```

These commands register `Python (off-shell angular coefficients, CPU)` and `Python (off-shell angular coefficients, GPU)`, respectively. Prefer the facility's `pixi` kernel when available because its environment choice is stored with the notebook.

### 5. Point the notebook to the LHE sample

In the configuration cell, change:

```python
LHE_FILE = Path("/path/visible/from/jupyter/events.lhe")
MAX_EVENTS = 20_000
```

Plain `.lhe` and gzip-compressed `.lhe.gz` files are accepted. Start with a modest `MAX_EVENTS`; after the object counts, recoil checks, and angular plots look correct, set it to `None` to process the full file.

The reader requires exactly one final-state particle for each of PDG IDs $11$, $-11$, $13$, and $-13$. It deliberately ignores all intermediate Higgs and $Z$ records. It defines $Z_1=p_{\mu^-}+p_{\mu^+}$, $Z_2=p_{e^-}+p_{e^+}$, and $H_{\mathrm{cand}}=Z_1+Z_2$, so the result is independent of whether IDs 23 and 25 appear in the LHE history.

### 6. Validate the runtime, then train

Before training, run this diagnostic cell:

```python
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("Compiled CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

For CPU execution, the Python path should contain `.pixi/envs/analysis/`, `torch.version.cuda` should be `None`, and CUDA availability should be `False`. For GPU execution, the path should contain `.pixi/envs/analysis-gpu/`, the compiled CUDA version should be non-`None`, and availability should be `True`.

If `nvidia-smi` cannot see a device, stop and relaunch JupyterLab with a GPU instance. If `nvidia-smi` sees a device but PyTorch does not, reselect `analysis-gpu` in the Pixi property inspector and restart the kernel. Lightning's `GPU available: False` and PyTorch's `pin_memory ... no accelerator` warning are expected in the CPU environment but indicate a configuration problem in the GPU environment.

Run the notebook through the Born-projection and angle-diagnostic cells. The changes in $m_{4\ell}$ and $y_{4\ell}$ and the projected $p_{T,4\ell}$ should be consistent with numerical precision.

Training is disabled by default so that an accidental **Run All** first performs only the inexpensive validation. After inspection, set:

```python
RUN_MODEL = True
LOAD_TRAINED_MODEL = False
```

Rerun the training and closure cells. To reuse files already written under `models/angular_ratio/`, keep `RUN_MODEL=True` and change `LOAD_TRAINED_MODEL=True`.

The estimator duplicates each fit event:

- numerator weight: $w_i t_{\alpha\beta,i}$;
- denominator weight: $w_i(1-t_{\alpha\beta,i})$.

The two class weights are independently normalized for `density_ratio_trainer`. The code retains $Z_t$ and $Z_{1-t}$ and restores the factor $Z_t/Z_{1-t}$ at inference. Omitting this correction biases the recovered conditional angular moment.

The BCE/KL construction requires a non-negative measure. The notebook stops if it finds negative nominal LHE weights. Do not simply take absolute values: negative-weight samples require a separately derived and validated signed-measure strategy.

### 7. Preserve reproducibility across AF sessions

JupyterLab pods are ephemeral. Keep code, `pixi.toml`, the validated `pixi.lock`, and small configuration changes in Git. Keep LHE samples and generated model/figure directories on appropriate persistent storage, not in the repository. On a new pod, clone or pull the project and rerun `pixi install -e analysis` for CPU execution or `pixi install -e analysis-gpu` for GPU execution.

## Local development

With Pixi installed on any compatible Linux system, the CPU workflow is:

```bash
cd offshell_angular_coefficients
pixi install -e analysis
pixi run -e analysis test
pixi run -e analysis lab
```

On a Linux host with a compatible NVIDIA driver and CUDA-capable GPU, use:

```bash
cd offshell_angular_coefficients
pixi install -e analysis-gpu
pixi run -e analysis-gpu test
pixi run -e analysis-gpu lab
```

The analysis package is intentionally small. Physics transformations live in tested Python functions rather than notebook-only cells, which makes sign conventions and future changes reviewable.

## Density-ratio caveats and validation plan

The original events are split into fit and evaluation samples before the weighted duplication. Use the untouched evaluation events for physics closure. The toolkit currently performs its own row-level holdout split, so the two weighted copies of one source event can enter opposite internal partitions; built-in overtraining plots can therefore be optimistic for this soft-label construction.

Before using a coefficient in the publication result:

- close every retained real and imaginary component against the direct weighted angular projection;
- validate angle signs on hand-checked phase-space points;
- repeat training across seeds and model architectures or use an ensemble;
- study finite-MC and model uncertainty, binning stability, and the effect of any event selection;
- use event-grouped cross-validation or statistically independent samples for final closure; and
- document any treatment of negative weights before applying it.

## References

- Born-like recoil map: [arXiv:2606.11083](https://arxiv.org/abs/2606.11083)
- Four-lepton angular terminology and kinematic variables: [arXiv:1208.4018](https://arxiv.org/abs/1208.4018)
- LHE reader: [`pylhe`](https://scikit-hep.org/pylhe/)
- Lorentz vectors: [`vector`](https://vector.readthedocs.io/)
- Density-ratio toolkit: [`iris-hep/nsbi-lhc-toolkit`](https://github.com/iris-hep/nsbi-lhc-toolkit)
- UChicago JupyterLab and Pixi: [US ATLAS AF documentation](https://usatlas.github.io/af-docs/uchicago/jupyter/)
