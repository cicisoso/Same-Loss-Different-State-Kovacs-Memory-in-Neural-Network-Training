# E10 optimizer resets: the decision rule of PREREG note 10:36 applied to k_pairs.py outputs. For each comparison, the ratios
# rho = reset / unreset of the smoothed maximum (10-150 steps) and of the area (0-300 steps) of the excursion, on set A
# (set B reported with it): retained, rho >= 0.8 for both; reduced but retained, 0.2 <= rho < 0.8 for both; removed,
# rho < 0.2 for both; otherwise reshaped. The lag of the maximum and the late offset are reported, not classified. A reset
# excursion whose maximum or area is negative is flagged as sign reversed (reported as such, not as a removed memory).
# Usage: python -I k_reset_rule.py <out.json> "<label>|<pairs.json>|<reset run:base>|<unreset run:base>" ...
import sys, json, os

out, specs = sys.argv[1], sys.argv[2:]


def find(d, name):   # pair keys carry the time origin ("run:base@t0")
    k = [k for k in d if k == name or k.startswith(name + "@")]
    return d[k[0]] if len(k) == 1 else None


def rule(rp, ra):
    if rp >= 0.8 and ra >= 0.8: return "retained"
    if 0.2 <= rp < 0.8 and 0.2 <= ra < 0.8: return "reduced but retained"
    if rp < 0.2 and ra < 0.2: return "removed"
    return "reshaped"


res = {}
for sp in specs:
    label, f, rs, un = sp.split("|")
    d = json.load(open(f)) if os.path.exists(f) else {}
    a, b = find(d, rs), find(d, un)
    if a is None or b is None:
        res[label] = {"status": "pending"}; print(label, "pending"); continue
    e = {"status": "ok", "reset": rs, "unreset": un}
    for S in ("A", "B"):
        x, y = a.get(f"stats_{S}", {}), b.get(f"stats_{S}", {})
        if not x or not y or None in (x.get("peak"), x.get("area"), y.get("peak"), y.get("area")): continue   # window not covered
        e[S] = {"rho_peak": x["peak"] / y["peak"], "rho_area": x["area"] / y["area"],
                "peak": [x["peak"], y["peak"]], "t_peak": [x["t_peak"], y["t_peak"]], "area": [x["area"], y["area"]],
                "late": [x["late"], y["late"]], "late_window": [x["late_window"], y["late_window"]]}
        e[S]["class"] = rule(e[S]["rho_peak"], e[S]["rho_area"])
        e[S]["sign_reversed"] = bool(x["peak"] < 0 or x["area"] < 0)   # Codex round 5: flag a reversed response, which the
                                                                        # registered rule would file under "removed"
        e[S]["negative_endpoints"] = [k for k in ("peak", "area") if x[k] < 0]   # round 6: name the reversed endpoint(s)
    if "A" not in e: e["status"] = "incomplete: registered windows not covered on set A"
    e["class"] = e.get("A", {}).get("class"); e["agree_B"] = (e["B"]["class"] == e["class"]) if ("A" in e and "B" in e) else None   # None: no comparison on B
    res[label] = e
    print(label, e["class"], "(sign reversed)" if e.get("A", {}).get("sign_reversed") else "", "| B", e.get("B", {}).get("class"),
          {S: {k: round(v, 3) for k, v in e[S].items() if k.startswith("rho")} for S in ("A", "B") if S in e},
          {S: [round(v, 4) for v in e[S]["peak"]] + [round(v, 2) for v in e[S]["area"]] for S in ("A", "B") if S in e}, flush=True)
json.dump(res, open(out, "w"), indent=1)
