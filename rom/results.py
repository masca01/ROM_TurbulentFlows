"""
CSV helpers shared by the studies.

update_row()   (from convergence_csv.py) — see below.
append_row()   append one row to a results CSV, header written when the file is new
               (the append-and-resume pattern of the region map and every later study).
rewrite_with_row()  read the whole CSV, add one row, rewrite it (re100_* scripts).
done_rows()    keys of the rows already in a CSV (resume logic).
LOWDATA_FIELDS column schema of lowdata_results.csv (run_lowdata.py stage B and
               run_lowdata_subsets.py write the same file).

convergence_csv.py:
Tiny helper to record the LAST convergence point of each dataset in one shared
CSV (written into the dataset folder).  The POD and the NN scripts each call
update_row() after their run and fill in only their own columns, so the runs
merge into a single row per dataset instead of overwriting each other.

Columns
    dataset        base name of the .mat file (row key)
    Re             Reynolds number
    POD95_n        snapshots at the last POD sweep point, POD truncated to the k
                   modes retaining >= 95% of the pool energy
    POD95_e_mean   POD unexplained-energy fraction (||.||F^2 ratio) there
    POD95_modes    that k (N_MODES)
    POD99_n / POD99_e_mean / POD99_modes    same, truncated at 99% energy
    NN_n           snapshots at the last NN sweep point
    NN_e_mean      NN relative-energy error (1 - Ek/100) there
    NN_Ek          NN energy captured Ek (%) there
    NN_latent      beta-VAE latent dimension

Any other column handed to update_row (e.g. POD90_*) is appended after these.
Legacy untagged POD_* columns (from before the 95/99 split) are read as POD95_*.
"""
import os, csv, datetime

FIELDNAMES = ["dataset", "Re",
              "POD95_n", "POD95_e_mean", "POD95_modes",
              "POD99_n", "POD99_e_mean", "POD99_modes",
              "NN_n", "NN_e_mean", "NN_Ek", "NN_latent"]

_LEGACY = {"POD_n": "POD95_n", "POD_e_mean": "POD95_e_mean", "POD_modes": "POD95_modes"}


def update_row(csv_path, dataset, updates):
    """Insert/merge the row for `dataset` with `updates` (a dict) and rewrite the
    CSV.  Existing columns filled by the other scripts are preserved."""
    rows = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                row = {}
                for k, v in r.items():
                    k = _LEGACY.get(k, k)             # POD_* (old runs) -> POD95_*
                    if v not in ("", None) or k not in row:
                        row[k] = v if v is not None else ""
                rows[row["dataset"]] = row

    row = rows.get(dataset, {"dataset": dataset})
    row["dataset"] = dataset
    for k, v in updates.items():
        row[k] = v
    rows[dataset] = row

    # preferred order first, then any extra columns in the order they appear
    fields = list(FIELDNAMES)
    for r in rows.values():
        for k in r:
            if k not in fields:
                fields.append(k)

    def _re(r):
        try:
            return float(r.get("Re") or 0)
        except ValueError:
            return 0.0
    ordered = sorted(rows.values(), key=lambda r: (_re(r), r["dataset"]))

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r.get(k, "") for k in fields})
    return csv_path


# ------------------------------ results CSVs of the later studies ------------------------------

# lowdata_results.csv (run_lowdata.py B_FIELDS)
LOWDATA_FIELDS = ["date", "dataset", "n_real", "sampling", "subset", "kind", "latent", "generator", "trunc", "K",
                  "energy_K_pct", "fraction", "epochs", "n_aug", "n_train", "pod_ceiling_Ek", "baseline_Ek",
                  "headroom", "quality_err", "predicted", "Ek", "gain", "detR", "status", "wall_s"]


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def append_row(path, fields, row, makedirs=False):
    """Append one row; write the header first if the file is new. Missing columns are blank."""
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def rewrite_with_row(path, fields, row, makedirs=False):
    """Read every row already in the CSV, add `row`, and rewrite the whole file."""
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = list(csv.DictReader(open(path, newline=""))) if os.path.exists(path) else []
    rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in fields})


def done_rows(path, keyfn):
    if not os.path.exists(path):
        return set()
    return {keyfn(r) for r in csv.DictReader(open(path, newline=""))}
