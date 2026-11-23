"""Run from the project root after projecting the two publisher archives."""
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from replay import compare

DATA = Path("data")
MODELS = json.loads(Path("design.json").read_text())
POLICIES = ("keep","random","within_question","median","always")
DIRECTIONS = {"accuracy_pp":(100,0), "tokens":(0,1), "utility_0.1":(100,-.0001),
              "utility_1":(100,-.001), "utility_10":(100,-.01)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pair(path, models, repeats):
    rows = defaultdict(dict)
    with path.open(newline="",encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            model = row["model"]
            if model not in models:
                continue
            q, rep = row["question"], int(row["replicate"])
            cell = rows[q,model]
            if rep in cell or row["correct"] not in ("0","1"):
                raise ValueError(f"duplicate replicate or invalid label at {q}, {model}, {rep}")
            cell[rep] = (int(row["tokens"]),int(row["correct"]))
    valid, excluded = {}, {}
    slots = set(range(repeats))
    for q in sorted({q for q,_ in rows}):
        pair = [rows[q,m] for m in models]
        if any(set(cell)!=slots for cell in pair):
            excluded[q]="incomplete slots"
        elif any(t<0 for cell in pair for t,_ in cell.values()):
            excluded[q]="negative token count"
        else:
            valid[q] = [[cell[r] for r in range(repeats)] for cell in pair]
    return valid,excluded


def threshold(values, prob):
    return sorted(values)[math.ceil(prob*len(values))-1]


def evaluate(qids, archive, cut, median):
    gates = {q:[int(t>cut) for t,_ in archive[q][0]] for q in qids}
    random_rate = fmean(fmean(w) for w in gates.values())
    cells = {(p,d):[] for p in POLICIES for d in DIRECTIONS}
    for q in qids:
        first, second = archive[q]
        weights = gates[q]
        m = len(weights)
        baselines = dict(keep=[0]*m, random=[random_rate]*m,
                         within_question=[fmean(weights)]*m,
                         median=[int(t>median) for t,_ in first], always=[1]*m)
        for p,baseline in baselines.items():
            delta = [a-b for a,b in zip(weights,baseline)]
            for d,(a,b) in DIRECTIONS.items():
                constant = -sum(w*a*y for w,(_,y) in zip(delta,first))
                values = [a*y+b*t for t,y in second]
                cells[p,d].append((delta,values,constant))
    full_first = fmean(t for q in qids for t,_ in archive[q][0])
    capped = fmean(min(t,cut+1) for q in qids for t,_ in archive[q][0])
    random_first = (1-random_rate)*full_first
    return dict(n=len(qids),call_fraction=random_rate,first_tokens=full_first,
        capped_first_tokens=capped, random_bypass_first_tokens=random_first,
        streaming_minus_bypass_first_tokens=capped-random_first, first_accuracy=fmean(
        fmean(y for _,y in archive[q][0]) for q in qids),
        fallback_accuracy=fmean(fmean(y for _,y in archive[q][1]) for q in qids),
        mixed_questions=sum(0<sum(w)<len(w) for w in gates.values()),
        comparisons={p:{d:compare(cells[p,d]) for d in DIRECTIONS} for p in POLICIES})


def main():
    result = {}
    for label,spec in MODELS.items():
        path = DATA/spec["file"]
        if digest(path)!=spec["sha256"]:
            raise ValueError(f"input digest mismatch: {path}")
        archive, exclusions = load_pair(path,spec["models"],spec["replicates"])
        order = sorted(archive,key=lambda q:hashlib.sha256(q.encode()).digest())
        n = len(order)//5
        if n < 1:
            raise ValueError("insufficient calibration questions")
        calibration, test = order[:n],order[n:]
        lengths = [t for q in calibration for t,_ in archive[q][0]]
        cut, med = threshold(lengths,.75),threshold(lengths,.5)
        result[label] = dict(spec=spec,calibration=calibration,excluded=exclusions,cut=cut,median=med,
                             evaluation=evaluate(test,archive,cut,med),full=evaluate(order,archive,cut,med))
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    files=["replay.py","study.py","design.json","PROTOCOL.md","test_replay.py"]
    (out/"execution.json").write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),
        source={name:digest(Path(name)) for name in files},result_sha256=digest(out/"results.json")),
        indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
