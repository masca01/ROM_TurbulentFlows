==============================================================================
  CODE/ — what each .py script does
  Project: 597 Special Topics — turbulent-flow ROM (beta-VAE vs POD)
==============================================================================

Folder layout (all scripts assume this):
    CODING/CODE/        <- the .py scripts (this folder)
    CODING/DATA/        <- input .mat datasets
    CODING/DATA/AUGMENTED/   <- POD-augmented bundles (.mat) produced here
    CODING/bestModels/  <- saved checkpoints (.pt) and POD files (.npz)

Most scripts have a CONFIG block near the top: edit DATA_FILE / file paths and
hyperparameters there, then run with:  python3 <script>.py


==============================================================================
  1. CORE TRAINING & BASELINE
==============================================================================

beta_vae.py
    THE base script. Trains a single beta-VAE (conv encoder + decoder) on one
    .mat dataset to build a reduced-order model of the flow.
    - Splits the input file ITSELF into train/validation (random, VAL_FRAC).
    - Loss = reconstruction MSE + BETA * KL.
    - Tracks validation Ek (% energy reconstructed) and det(R) (latent
      disentanglement).
    - Saves the best-val checkpoint to ../bestModels/model_betaVAE_*.pt and
      shows learning curves + a TRUE | RECON | ERROR field figure.
    - Provides shared building blocks (load_data, Encoder, Decoder, vae_loss,
      val_loss, compute_ek, compute_det_R) that the other scripts import.
    Use it when: you want a plain VAE trained on one real dataset.

pod.py
    Classical POD (Proper Orthogonal Decomposition) baseline, via SVD.
    Computes spatial modes + temporal coefficients, evaluates how much energy
    is captured at several ranks (RANKS = [1,5,20,50]), and saves a pod_*.npz
    to ../bestModels/. This is the linear ROM that the VAE is compared against.
    Use it when: you want the POD reference / energy-vs-rank curves.


==============================================================================
  2. LOSS / ARCHITECTURE VARIANTS OF THE VAE
==============================================================================

beta_vae_bsp.py
    beta-VAE + Binned Spectral Power (BSP) loss (Chakraborty, Mohan & Maulik
    2026, arXiv:2502.00472v3). Adds a frequency-domain term that matches the
    binned energy spectra of reconstruction vs truth, so SMALL SCALES (high
    wavenumbers) are reproduced better (fights spectral bias).
        Loss = MSE + BETA*KL + MU * L_BSP   (bins weighted ~k^2, Sobolev-like)
    Single network. Same save/plot behaviour as beta_vae.py.
    Use it when: the standard VAE smooths out fine-scale structure.

beta_vae_multiscale.py
    Splits the field into LARGE and SMALL scales with a circular spectral
    filter at cutoff K_CUT, trains ONE beta-VAE per scale, then reconstructs
    as (large + small) and compares to a pre-trained full-field VAE.
    Specific to the BOX/isotropic dataset (uses u, w).
    Use it when: you want to test scale-separated VAEs (2 bands).

beta_vae_nbands.py
    Generalisation of the multiscale script to N spectral bands. Given a list
    of cutoffs K_CUTS, partitions the field into N circular bands, trains one
    beta-VAE per band (each with its own latent dim), sums the reconstructions
    and compares to the full-field VAE. BOX/isotropic dataset (u, w).
    Use it when: you want more than 2 frequency bands.


==============================================================================
  3. POD-BASED DATA AUGMENTATION  (pretraining experiment)
==============================================================================
Idea: fit POD on the TRAINING snapshots only, then synthesise extra snapshots
by resampling the POD temporal coefficients. Each generator below writes a
bundle to ../DATA/AUGMENTED/ containing:
        train_real, train_aug, val_real   (val_real is ALWAYS real data)
All three share the same RNG_SEED / VAL_FRAC / N_AUG, so their train_real and
val_real splits are identical -> the only difference is HOW train_aug is made.

pod_augment_gaussian.py   (STRATEGY A — "gaussian")
    Each mode's coefficient drawn INDEPENDENTLY: a_i ~ N(mean_i, std_i).
    Simplest baseline. Breaks phase coupling between modes (can be
    non-physical for periodic flows like the cylinder).

pod_augment_highmodes.py  (STRATEGY B — "highmodes")
    Keeps the leading N_LEAD coherent modes from a REAL snapshot (preserves
    large-scale phase, e.g. vortex shedding) and resamples only the higher,
    small-scale modes from Gaussians. Targets small-scale augmentation safely.

pod_augment_noise.py      (STRATEGY C — labelled "joint")
    Bootstrap: take a real snapshot's coefficient vector and add small jitter,
    eps_i ~ N(0, (NOISE_LEVEL*std_i)^2). Preserves the joint/phase structure
    empirically. NOTE: filename says "noise", the STRATEGY tag inside is
    "joint" — same thing, this is strategy C.

pod_augment_gmm.py        (ASSESSMENT CONFIG #3 — "gmm")
    Joint-density sampling: fits a Gaussian Mixture Model on the leading
    N_JOINT modes JOINTLY (keeps phase coupling / attractor shape), Gaussian
    marginals on the tail, and screens samples by total modal energy.
    Saves .npz instead of .mat when arrays are too big (full 2-plates).

pod_augment_galerkin.py   (ASSESSMENT CONFIG #4 — "galerkin")
    Unclosed POD-Galerkin ROM by system identification (SINDy-style):
    estimates da/dt by central finite differences on the training coefficient
    series, fits  da_i/dt = c_i + L_ij a_j + Q_ijk a_j a_k  by ridge least
    squares (linear in c, L, Q), then integrates SHORT synthetic trajectories
    with RK4 from perturbed real initial conditions, truncating any that
    leave the observed energy band (the rejection rate measures how badly
    the unclosed model drifts -> motivates config #5 closure).
    MODEL="auto": the da/dt (ODE) fit needs >~8 snapshots per oscillation
    period (2-plates: ~16, OK). Coarser data (cylinder TensorRe160: ~4) is
    handled by fitting the DISCRETE one-step map a(n)->a(n+1) instead — no
    derivatives needed — and iterating it. If the generated trajectories keep
    leaving the observed energy band, the script backs off N_ROM
    automatically until the trajectories are healthy.

pod_augment_galerkin_ns.py   ("galerkin_ns")
    Same Galerkin ROM family as pod_augment_galerkin.py, but the model
    coefficients are derived ANALYTICALLY by projecting the Navier-Stokes
    convection and diffusion operators onto the POD modes (a Python port of
    Prof. Dawson's ExampleCode/nseGalerkinCoeffsDemo.m), instead of being
    identified from data. Needs RE and the grid spacing (read from
    DataX/DataY; 2-plates only for now, C=2 velocity fields required), and
    rescales the POD modes to be orthonormal in the physical L2 inner
    product. Generation/screening identical to the identified version, so
    the two bundles differ ONLY in where c/L/Q come from — physics-projected
    vs data-identified. Prints a per-mode "derivative R^2" diagnostic
    (projected model vs observed central differences; low values = closure
    error, nothing is fitted).

    NOTE: both pod_augment_galerkin.py and pod_augment_galerkin_ns.py are
    self-contained — they inline their own load_data / POD / bundle helpers
    and do NOT import from beta_vae.py, so they can be read and run on their
    own (this is the version commented for hand-in).

    (Set DATA_FILE in each generator to the dataset you want to augment;
     remember the trailing ".mat".)


==============================================================================
  4. TRAINING / TESTING ON AUGMENTED DATA
==============================================================================

test_augmented_vae.py
    LIGHT tester. Loads one augmented bundle, trains the identical beta-VAE on
    train_real (+ train_aug if USE_AUG=True), validates on val_real ONLY, and
    appends best Ek / det(R) to augment_results.csv. Does NOT save a model or
    make figures. Run it once per strategy (+ once with USE_AUG=False for the
    real-only baseline) to compare strategies in the CSV.

train_augmented_vae.py
    FULL version of the above. Same leak-free setup (train on augmented,
    validate on REAL only), but ALSO:
      - saves the best-val checkpoint to ../bestModels/model_augVAE_*.pt
        (self-contained: val_real + normalization baked in, so you can
         re-visualize without reloading the big .mat), and
      - shows beta_vae.py-style figures (learning curves +
        TRUE | RECON | ERROR for a real validation snapshot), and
      - appends the same row to augment_results.csv.
    Use this one when you want the saved model + plots, not just CSV numbers.
    The Ek in the figure matches the Ek in the CSV (same model, same real val).


==============================================================================
  5. VISUALIZATION / ANALYSIS OF SAVED MODELS
==============================================================================

inspect_vae.py
    Reloads a saved beta-VAE checkpoint (model_betaVAE_*.pt from beta_vae.py)
    and reproduces its figures WITHOUT retraining: learning curves and a
    TRUE | RECON | ERROR field at a chosen snapshot. Point VAE_FILE at the .pt.
    (Built for beta_vae.py checkpoints; reloads the original .mat via val_idx.)

view_augmented_movie.py
    "Video" sanity check of any augmentation bundle: plays REAL vs SYNTHETIC
    snapshots side by side (shared color scale) with a fluctuation-energy
    trace of both series underneath. For dynamical strategies (galerkin) the
    synthetic frames should evolve smoothly like a flow; for statistical ones
    (gmm/joint/gaussian) frames are independent draws. Set AUG_FILE, COMP;
    SAVE_PATH writes an .mp4/.gif instead of the interactive window.

analysis.py
    Side-by-side comparison of POD modes vs beta-VAE modes. Loads both a
    model_betaVAE_*.pt and a pod_*.npz, plots the dominant spatial modes of
    each method and compares their reconstructions/spectra. The "which method
    wins" figure. Set VAE_FILE, POD_FILE, DATA_FILE in its CONFIG.


==============================================================================
  TYPICAL WORKFLOWS
==============================================================================

A) Plain VAE vs POD on one dataset:
     1) python3 pod.py            (set DATA_FILE)   -> pod_*.npz
     2) python3 beta_vae.py       (set DATA_FILE)   -> model_betaVAE_*.pt
     3) python3 analysis.py       (point at both)   -> comparison figures
     (re-view a model later: python3 inspect_vae.py)

B) POD data-augmentation comparison:
     1) python3 pod_augment_gaussian.py
        python3 pod_augment_highmodes.py
        python3 pod_augment_noise.py        -> 3 bundles in DATA/AUGMENTED/
     2) For each bundle, set AUG_FILE in train_augmented_vae.py and run:
          - USE_AUG=True  for the augmented run
          - USE_AUG=False once for the real-only baseline
        -> checkpoints + figures, and rows in augment_results.csv
     3) Compare the best_val_Ek column across rows (higher = better).
        (Use test_augmented_vae.py instead if you only want the CSV numbers.)

C) Better small scales:
     - beta_vae_bsp.py        (spectral loss), or
     - beta_vae_multiscale.py / beta_vae_nbands.py (scale-separated VAEs).
==============================================================================
