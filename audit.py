"""Compare two cascade policies from aligned positional marginal logs."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from replay import compare


def load_cells(path, quality_weight, token_weight):
    questions = defaultdict(dict)
    with path.open(encoding="utf-8",newline="") as fh:
        for row in csv.DictReader(fh):
            q, index = row["question"],int(row["replicate"])
            g,h,y,z,t = (float(row[k]) for k in ("gate","comparator","first_grade","fallback_grade","fallback_tokens"))
            if not q or index<0 or not all(0<=x<=1 for x in (g,h,y,z)) or not math.isfinite(t) or t<0:
                raise ValueError(f"invalid row for {q!r}, slot {index}")
            if index in questions[q]:
                raise ValueError(f"duplicate slot {index} for {q}")
            questions[q][index] = g-h,quality_weight*z+token_weight*t,-quality_weight*(g-h)*y
    cells=[]
    for q,slots in sorted(questions.items()):
        if set(slots)!=set(range(len(slots))):
            raise ValueError(f"noncontiguous replicate indices for {q}")
        w,v,c=zip(*(slots[i] for i in range(len(slots))))
        cells.append((w,v,sum(c)))
    return cells


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--quality-weight",type=float,default=100.)
    parser.add_argument("--token-weight",type=float,default=0.)
    args=parser.parse_args()
    if not all(math.isfinite(x) for x in (args.quality_weight,args.token_weight)):
        parser.error("weights must be finite")
    print(json.dumps(compare(load_cells(args.input,args.quality_weight,args.token_weight)),
                     indent=2,allow_nan=False))
