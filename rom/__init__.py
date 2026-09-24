"""
rom — shared library of the MMAE 597 beta-VAE / POD-Galerkin study.

Nothing runs on import. The modules are imported explicitly, e.g.

    from rom import paths, data, split, pod, vae, augment, results
    from rom import galerkin_ns as gns, galerkin_data as gd

    paths          DATA / AUGMENTED / MODELS / RESULTS folders
    data           dataset registry and .mat loaders
    split          validation split, subset draws, convergence file names
    pod            POD by the method of snapshots, ceilings, truncation rules
    vae            beta-VAE network, metrics and the shared training loop
    galerkin_ns    NS-projected POD-Galerkin model (operators l, q), RK4, screening
    galerkin_data  data-identified POD-Galerkin model
    augment        generator settings and the augmentation arms
    results        CSV helpers (append / resume / convergence last point)
"""
