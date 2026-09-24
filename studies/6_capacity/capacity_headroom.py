"""
CAPACITY-CORRECTED HEADROOM.

headroom as used so far = POD ceiling of the real subset - real-only Ek of the network.
That mixes two different limits: how much of the flow the SNAPSHOTS span (POD ceiling) and how
much of it the NETWORK can represent at all (its latent dimension). When the latent space is the
binding constraint, part of that headroom cannot be filled by data of any kind.

This script reports both. The capacity ceiling of a (flow, latent) pair is measured, not assumed:
it is the Ek the same network reaches when data are no longer the constraint, taken from the
convergence study (full training pool) and, for Re 100 with latent 5, from the largest subsets of
the tolerance/modes grid. Those numbers are lower bounds - the curves are still rising slowly - so
the corrected headroom is conservative.

    corrected headroom = min(POD ceiling, capacity ceiling) - real-only Ek

Output: ../../convergence/headroom_corrected.csv, one row per augmented training, and a summary
of how many cells and how many screening decisions actually change.
"""
import csv, os, statistics as st, collections

from rom import paths

CONV = paths.RESULTS
OUT = os.path.join(CONV, "headroom_corrected.csv")
OUT_CELLS = os.path.join(CONV, "headroom_corrected_cells.csv")

# Ek reached with the full training pool, from Alpha0_convergence.xlsx / Data2PlatesGap1Re100_convergence.xlsx
# (random 10% split, same recipe), and from re100_tol_modes_grid.csv for Re 100 at latent 5.
CAPACITY = {
    ("Re50", 5): 98.3, ("Re60", 7): 98.1, ("Re70", 9): 98.1, ("Re80", 11): 98.9,
    ("Re100", 15): 90.0, ("Re100", 5): 47.9, ("Re100", 25): 90.0,
}
SOURCE = {
    ("Re50", 5): "convergence study, 1350 snapshots",
    ("Re60", 7): "convergence study, 1350 snapshots",
    ("Re70", 9): "convergence study, 1350 snapshots",
    ("Re80", 11): "convergence study, 1350 snapshots",
    ("Re100", 15): "convergence study, 900 snapshots",
    ("Re100", 5): "tolerance/modes grid, 775 snapshots (still rising)",
    ("Re100", 25): "lower bound: latent 15 value, capacity cannot be smaller",
}
MEASURED = os.path.join(CONV, "capacity_ceilings.csv")   # written by run_capacity.py, if it was run
FILES = [("region_map_results.csv", "fid_err"), ("experiments_results.csv", "quality_err"),
         ("experiments2_results.csv", "quality_err"), ("lowdata_results.csv", "quality_err")]
RULE_ERR, RULE_HEAD = 0.5, 10.0
FIELDS = ["source_file", "dataset", "latent", "n_real", "generator", "modes", "fraction", "subset",
          "pod_ceiling_Ek", "capacity_Ek", "effective_ceiling", "baseline_Ek", "headroom",
          "headroom_corrected", "rom_error", "gain", "passes_old", "passes_new"]


def rows(name):
    return list(csv.DictReader(open(os.path.join(CONV, name), newline="")))


def load_measured():
    """Replace the built-in ceilings by directly measured ones where run_capacity.py has produced
    them (same recipe, full training pool). Measured values win over the table."""
    if not os.path.exists(MEASURED):
        return 0
    n = 0
    for r in csv.DictReader(open(MEASURED, newline="")):
        try:
            key, ek = (r["dataset"], int(r["latent"])), float(r["Ek"])
        except (KeyError, ValueError):
            continue
        CAPACITY[key] = ek
        SOURCE[key] = f"measured on the full pool ({r['n_train']} snapshots, {r['epochs']} epochs)"
        n += 1
    return n


def main():
    n_measured = load_measured()
    if n_measured:
        print(f"{n_measured} capacity ceilings read from {os.path.basename(MEASURED)} "
              f"(these override the built-in table)\n")
    out, changed = [], 0
    for fname, ekey in FILES:
        for r in rows(fname):
            if r.get("kind") != "aug" or r.get("generator") not in ("galerkin", "galerkin_ns"):
                continue
            if not (r.get(ekey) and r.get("headroom") and r.get("baseline_Ek")):
                continue
            ds, lat = r["dataset"], int(r["latent"])
            cap = CAPACITY.get((ds, lat))
            ceiling = float(r["pod_ceiling_Ek"])
            base = float(r["baseline_Ek"])
            eff = min(ceiling, cap) if cap is not None else ceiling
            head_old = float(r["headroom"])
            head_new = eff - base
            err = float(r[ekey])
            if abs(head_new - head_old) > 0.05:
                changed += 1
            out.append(dict(
                source_file=fname, dataset=ds, latent=lat, n_real=int(r["n_real"]), generator=r["generator"],
                modes=r.get("mode_level") or r.get("trunc", ""), fraction=r.get("fraction", ""),
                subset=r.get("subset", ""), pod_ceiling_Ek=round(ceiling, 2),
                capacity_Ek=cap if cap is not None else "", effective_ceiling=round(eff, 2),
                baseline_Ek=round(base, 2), headroom=round(head_old, 2), headroom_corrected=round(head_new, 2),
                rom_error=round(err, 4), gain=round(float(r["gain"]), 3),
                passes_old=int(err <= RULE_ERR and head_old >= RULE_HEAD),
                passes_new=int(err <= RULE_ERR and head_new >= RULE_HEAD)))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} augmented trainings written to {OUT}")
    print(f"{changed} of them have a different headroom once capacity is taken into account\n")

    # ---- per cell, and what the correction does to the screen
    cells = collections.defaultdict(lambda: {"g": [], "ho": [], "hn": [], "e": [], "old": [], "new": []})
    for r in out:
        k = (r["source_file"], r["dataset"], r["n_real"], r["latent"], r["generator"], r["modes"],
             r["fraction"])
        d = cells[k]
        d["g"].append(r["gain"]); d["ho"].append(r["headroom"]); d["hn"].append(r["headroom_corrected"])
        d["e"].append(r["rom_error"]); d["old"].append(r["passes_old"]); d["new"].append(r["passes_new"])
    def tab(key):
        tp = sum(1 for d in cells.values() if st.mean(d[key]) > 0.5 and st.mean(d["g"]) >= 1)
        fp = sum(1 for d in cells.values() if st.mean(d[key]) > 0.5 and st.mean(d["g"]) < 1)
        fn = sum(1 for d in cells.values() if st.mean(d[key]) <= 0.5 and st.mean(d["g"]) >= 1)
        tn = len(cells) - tp - fp - fn
        return tp, fp, fn, tn
    for key, lab in (("old", "headroom as published "), ("new", "capacity-corrected     ")):
        tp, fp, fn, tn = tab(key)
        print(f"{lab}: screen says gain {tp + fp:2d} cells ({tp} gained, {fp} did not), "
              f"says no gain {fn + tn:2d} ({fn} gained) → correct {tp + tn}/{len(cells)}")
    moved = [(k, st.mean(d['ho']), st.mean(d['hn']), st.mean(d['g']))
             for k, d in cells.items() if abs(st.mean(d["ho"]) - st.mean(d["hn"])) > 0.05]
    print(f"\ncells whose headroom changes: {len(moved)} of {len(cells)}")
    for k, ho, hn, g in sorted(moved, key=lambda x: -abs(x[1] - x[2])):
        print(f"   {k[1]} latent {k[3]}, {k[2]} real, {k[5]} modes, f={k[6]}: "
              f"headroom {ho:.1f} → {hn:.1f}   (gain {g:+.1f})")
    # ---- per-cell file, the one every figure should read
    cfields = ["source_file", "dataset", "latent", "n_real", "generator", "modes", "fraction", "subsets",
               "rom_error", "headroom", "headroom_corrected", "capacity_Ek", "mean_gain", "ci_half",
               "improved", "passes_old", "passes_new"]
    import math
    with open(OUT_CELLS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cfields)
        w.writeheader()
        for k, d in cells.items():
            n = len(d["g"])
            half = (4.303 if n == 3 else 2.776 if n == 5 else 2.365 if n == 8 else 2.262) * \
                   (st.stdev(d["g"]) / math.sqrt(n)) if n > 1 else float("nan")
            mean = st.mean(d["g"])
            w.writerow(dict(source_file=k[0], dataset=k[1], latent=k[3], n_real=k[2], generator=k[4],
                            modes=k[5], fraction=k[6], subsets=n, rom_error=round(st.mean(d["e"]), 4),
                            headroom=round(st.mean(d["ho"]), 2), headroom_corrected=round(st.mean(d["hn"]), 2),
                            capacity_Ek=CAPACITY.get((k[1], int(k[3])), ""), mean_gain=round(mean, 3),
                            ci_half=("" if n < 2 else round(half, 3)),
                            improved=int(n > 1 and mean >= 1 and mean - half > 0),
                            passes_old=int(st.mean(d["old"]) > 0.5), passes_new=int(st.mean(d["new"]) > 0.5)))
    print(f"\nper-cell summary written to {OUT_CELLS} ({len(cells)} cells)")

    print("\ncapacity ceilings used:")
    for k, v in sorted(CAPACITY.items()):
        print(f"   {k[0]:6s} latent {k[1]:2d}: {v:5.1f}%   [{SOURCE[k]}]")


if __name__ == "__main__":
    main()
