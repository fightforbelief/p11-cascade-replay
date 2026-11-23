"""Post hoc, question-level sensitivity audit. Run from the pinned P11 checkout."""
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

N_BOOT = 5000
N_SPLIT = 200


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def central(values):
    v = sorted(values)
    def quantile(p):
        h = p*(len(v)-1)
        return v[math.floor(h)]*(1-h%1)+v[math.ceil(h)]*(h%1)
    return [round(quantile(p),5) for p in (.025,.5,.975)]


def question_rows(data, test, cut, use_chars):
    out=[]
    for q in test:
        first, second = data[q]
        m=len(first)
        flags=[row[2 if use_chars else 1]>cut for row in first]
        rate=sum(flags)/m
        y=statistics.mean(r[0] for r in first)
        z=statistics.mean(r[0] for r in second)
        t=statistics.mean(r[1] for r in second)
        kept_loss=sum(g*r[0] for g,r in zip(flags,first))/m
        out.append((rate,rate*z-kept_loss,z-y,rate*t,t,flags,[r[0] for r in second],kept_loss))
    return out


def contrast(rows):
    rate, gain, gap, cost, tok = (sum(col)/len(rows) for col in zip(*(r[:5] for r in rows)))
    return (100*(gain-rate*gap),cost-rate*tok)


def n_reversals(rows):
    m=len(rows[0][5])
    rate=sum(r[0] for r in rows)/len(rows)
    base=sum(r[7]+rate*r[2] for r in rows)
    matrix=[[sum(row[5][i]*row[6][j] for row in rows)/m for j in range(m)] for i in range(m)]
    vals=[sum(matrix[i][j] for i,j in enumerate(p))-base for p in itertools.permutations(range(m))]
    return sum(v < -1e-10 for v in vals),sum(v > 1e-10 for v in vals)


def audit(spec,char,expected):
    path="data/"+spec["file"]
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==spec["sha256"]
    chars={(r["question"],r["model"],int(r["replicate"])):int(r["characters"])
           for r in read_csv("data/aime_lengths.csv")} if char else {}
    cells=defaultdict(dict)
    for row in read_csv(path):
        q,model,i=row["question"],row["model"],int(row["replicate"])
        cells[q,model][i]=(int(row["correct"]),int(row["tokens"]),chars.get((q,model,i)))
    qs=sorted({q for q,model in cells})
    data={q:[list(cells[q,model][i] for i in range(spec["replicates"]))
             for model in spec["models"]] for q in qs}
    def split(cal):
        vals=sorted(row[2 if char else 1] for q in cal for row in data[q][0])
        cut=vals[math.ceil(.75*len(vals))-1]
        return question_rows(data,[q for q in qs if q not in cal],cut,char)
    nc=len(qs)//5
    original=split(set(sorted(qs,key=lambda q:hashlib.sha256(q.encode()).digest())[:nc]))
    point=contrast(original)
    for val,key in zip(point,("accuracy_pp","tokens")):
        assert abs(expected["comparisons"]["random"][key]["product"]-val)<1e-7
    rng=random.Random(20261123)
    boots=[contrast(rng.choices(original,k=len(original))) for _ in range(N_BOOT)]
    rng=random.Random(20261124)
    alt=[];both=0;negative=0
    for _ in range(N_SPLIT):
        rows=split(set(rng.sample(qs,nc)))
        alt.append(contrast(rows))
        if spec["replicates"]==4:
            neg,pos=n_reversals(rows)
            both+=int(neg>0 and pos>0)
            negative+=neg
    loo=[contrast(original[:i]+original[i+1:])[0] for i in range(len(original))]
    return dict(product=point,bootstrap_accuracy=central([v[0] for v in boots]),
                bootstrap_fallback_tokens=central([v[1] for v in boots]),
                leave_one_out_accuracy=[min(loo),max(loo)],split_accuracy=central([v[0] for v in alt]),
                split_fallback_tokens=central([v[1] for v in alt]),split_positive=sum(v[0]>0 for v in alt),
                original_global_sign_counts=n_reversals(original) if spec["replicates"]==4 else None,
                resplits_with_both_global_signs=both if spec["replicates"]==4 else None)


def main():
    specs=json.loads(Path("design.json").read_text())
    base=json.loads(Path("reports/results.json").read_text())
    sens=json.loads(Path("reports/aime_characters.json").read_text())
    report={}
    for name,s in specs.items():
        report[name]=audit(s,False,base[name]["evaluation"])
        if name in sens:
            report[name+"-characters"]=audit(s,True,sens[name]["evaluation"])
    Path("reports/question_sensitivity.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")


if __name__=="__main__":
    main()
