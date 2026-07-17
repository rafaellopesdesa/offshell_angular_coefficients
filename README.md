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
4. For a first functional test, request a few CPU cores, sufficient memory for the intended LHE sample, and no GPU. The density-ratio training can be moved to a GPU session later if profiling justifies it.
5. Select `ml_platform:latest`, the facility's recommended ML image, and launch the server.

The current resource limits and launch options are documented by the [UChicago Analysis Facility](https://usatlas.github.io/af-docs/uchicago/jupyter/). They can change, so use that page as the authoritative source.

### 2. Clone the private repository

Open a JupyterLab terminal and clone with either an SSH key registered at GitHub or an authenticated HTTPS session:

```bash
git clone git@github.com:rafaellopesdesa/offshell_angular_coefficients.git
cd offshell_angular_coefficients
```

For HTTPS, use the repository's GitHub clone URL and the facility-supported credential flow. Do not put a personal access token in a notebook or committed file.

### 3. Create the project environment

Pixi is provided by the `ml_platform` image. From the repository root, run:

```bash
pixi install -e analysis
pixi run -e analysis test
```

The first command resolves and installs the packages declared in `pixi.toml`. The second runs the unit tests. A correct setup currently reports seven passing tests. The environment includes `pylhe`, `vector`, `mplhep`, SciPy, pandas, PyTorch/Lightning, ONNX Runtime, Jupyter support, and a commit-pinned build of `nsbi-common-utils`.

The toolkit wheel is stored under `vendor/` and was built from commit `fc09848fc6540fd32310faebbe9db6eea7ecd17b` of `iris-hep/nsbi-lhc-toolkit`. This avoids checking out the upstream repository during the Pixi solve: that repository contains Git LFS rules for large example files, while the analysis needs only its small pure-Python package. The wheel's provenance, rebuild command, and SHA-256 digest are recorded in `vendor/README.md`.

The first successful solve also creates `pixi.lock`. Commit that lock file when the environment has been validated at the AF; subsequent sessions can then reconstruct the exact solve rather than only respecting the version constraints in `pixi.toml`.

### 4. Select the Pixi kernel

Open `notebooks/01_lhe_angular_coefficients.ipynb`, then follow the facility's `pixi-kernel` workflow:

1. Click the kernel selector in the upper-right corner and select **pixi**.
2. Open the property inspector using the gear icon in the right sidebar.
3. Select the `analysis` environment associated with this project's `pixi.toml`.
4. Save the notebook.
5. Restart the kernel.

If the environment is not listed, confirm that the notebook was opened from the cloned project and run:

```bash
pixi install -e analysis
pixi list -e analysis pixi-kernel
```

Then restart JupyterLab and repeat the selection. The [facility Pixi instructions](https://usatlas.github.io/af-docs/uchicago/jupyter/#pixi) contain the current UI workflow and troubleshooting advice.

As a terminal-only fallback, the project also defines:

```bash
pixi run -e analysis kernel
```

This registers a conventional user kernel named `Python (off-shell angular coefficients)`. Prefer the facility's `pixi` kernel when available because its environment choice is stored with the notebook.

### 5. Point the notebook to the LHE sample

In the configuration cell, change:

```python
LHE_FILE = Path("/path/visible/from/jupyter/events.lhe")
MAX_EVENTS = 20_000
```

Plain `.lhe` and gzip-compressed `.lhe.gz` files are accepted. Start with a modest `MAX_EVENTS`; after the object counts, recoil checks, and angular plots look correct, set it to `None` to process the full file.

The reader requires exactly one final-state particle for each of PDG IDs $11$, $-11$, $13$, and $-13$. It deliberately ignores all intermediate Higgs and $Z$ records. It defines $Z_1=p_{\mu^-}+p_{\mu^+}$, $Z_2=p_{e^-}+p_{e^+}$, and $H_{\mathrm{cand}}=Z_1+Z_2$, so the result is independent of whether IDs 23 and 25 appear in the LHE history.

### 6. Validate first, then train

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

JupyterLab pods are ephemeral. Keep code, `pixi.toml`, the validated `pixi.lock`, and small configuration changes in Git. Keep LHE samples and generated model/figure directories on appropriate persistent storage, not in the repository. On a new pod, clone or pull the project and rerun `pixi install -e analysis`.

## Local development

With Pixi installed on any compatible Linux system:

```bash
cd offshell_angular_coefficients
pixi install -e analysis
pixi run -e analysis test
pixi run -e analysis lab
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
