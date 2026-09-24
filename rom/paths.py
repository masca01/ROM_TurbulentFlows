"""
Single source for the folders the code reads and writes.

Every folder is anchored at the repository root (the folder that holds rom/),
so the resolved paths do not depend on where a script lives or on the working
directory:

    DATA       <repo>/../DATA              input .mat datasets
    AUGMENTED  DATA/AUGMENTED              bundles and convergence .mat files
    MODELS     <repo>/../bestModels        checkpoints (.pt) and POD files (.npz)
    RESULTS    <repo>/../../convergence    every results CSV / xlsx / figure

Each can be overridden with an environment variable (used by the smoke tests
to point the code at fake data):

    ROM_DATA_DIR, ROM_MODELS_DIR, ROM_RESULTS_DIR

Nothing is created on import; the scripts create the folders they write to.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dir(env, default):
    value = os.environ.get(env)
    if value:
        return os.path.normpath(os.path.abspath(os.path.expanduser(value)))
    return default


DATA      = _dir("ROM_DATA_DIR", os.path.normpath(os.path.join(REPO_ROOT, "..", "DATA")))
AUGMENTED = os.path.join(DATA, "AUGMENTED")
MODELS    = _dir("ROM_MODELS_DIR", os.path.normpath(os.path.join(REPO_ROOT, "..", "bestModels")))
RESULTS   = _dir("ROM_RESULTS_DIR", os.path.normpath(os.path.join(REPO_ROOT, "..", "..", "convergence")))
