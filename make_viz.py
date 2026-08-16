"""Generate the self-contained visualizer HTML (solve playback + training telemetry).

Embeds solve traces, decimated training metrics, and eval reports as JSON.
Re-run any time to refresh; the page has no external dependencies.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def prep_metrics(path, max_points=1200):
    recs = load_jsonl(path)
    loss = [(r["iter"], r["loss"]) for r in recs if "loss" in r]
    if len(loss) > max_points:
        stride = len(loss) // max_points + 1
        loss = loss[::stride]
    probes = [r for r in recs if "probe" in r]
    depths = sorted({d for r in probes for d in r["probe"]}, key=int)
    probe_series = {d: [(r["iter"], r["probe"].get(d)) for r in probes] for d in depths}
    return {"loss": loss, "probe": probe_series}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "viz", "index.html"))
    ap.add_argument("--traces2"); ap.add_argument("--metrics2"); ap.add_argument("--report2")
    ap.add_argument("--traces3"); ap.add_argument("--metrics3"); ap.add_argument("--report3")
    ap.add_argument("--meta", default="{}")
    args = ap.parse_args()

    data = {"meta": json.loads(args.meta), "cubes": {}}
    for tag, tr, mt, rp in (("2x2", args.traces2, args.metrics2, args.report2),
                            ("3x3", args.traces3, args.metrics3, args.report3)):
        entry = {}
        if tr and os.path.exists(tr):
            entry["traces"] = json.load(open(tr))
        if mt and os.path.exists(mt):
            entry["metrics"] = prep_metrics(mt)
        if rp and os.path.exists(rp):
            entry["report"] = json.load(open(rp))
        if entry:
            data["cubes"][tag] = entry

    tpl = open(os.path.join(HERE, "viz_template.html")).read()
    html = tpl.replace("__DATA_JSON__", json.dumps(data))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
