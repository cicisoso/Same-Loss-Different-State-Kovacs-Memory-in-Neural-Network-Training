# Registered contest per case: pass flag and height ratio / time ratio / NRMSE of every model on every registered
# (non-exploratory) Kovacs case, from the scored contest (k_score.py output collected in k_contest.json).
# Usage: python -I contest_grid.py <k_contest.json>
import sys, json
c = json.load(open(sys.argv[1]))
W = [x for x in c["runs"] if x["kind"] != "exploratory"]
M = ["tissue", "mpl", "fsl", "nqm2", "nqm2pq", "nqmS"]
print("run".ljust(18), " ".join(m.ljust(21) for m in M))
for r in sorted({x["run"] for x in W}):
    row = []
    for m in M:
        x = [y for y in W if y["run"] == r and y["model"] == m][0]
        s = "%+.2f/%.2f/%.2f" % (x["height_ratio"], x["time_ratio"], x["nrmse"])
        row.append(("P " if x["pass"] else ". ") + s.ljust(19))
    print(r.ljust(18), " ".join(row))
