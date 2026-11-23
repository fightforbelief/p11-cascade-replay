"""Post-audit sensitivity. AIME gates use length of the returned answer string."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from project_input import AIME_MODELS,AIME_SHA,verify


def project(path):
    import pyarrow.parquet as pq
    verify(path,"sha256",AIME_SHA)
    records=[]
    counts=Counter()
    cols=["model_config","problem_idx","idx_answer","answer","output_tokens","correct"]
    for batch in pq.ParquetFile(path).iter_batches(columns=cols):
        for x in batch.to_pylist():
            model=AIME_MODELS.get(x["model_config"])
            if not model:
                continue
            if not isinstance(x["answer"],str):
                raise ValueError("missing answer")
            length=len(x["answer"])
            records.append((str(x["problem_idx"]),model,x["idx_answer"],length))
            if length and x["output_tokens"]==0:
                counts[model,"nonempty_zero"]+=1
                counts[model,"correct_nonempty_zero"]+=int(x["correct"])
    with Path("data/aime_lengths.csv").open("w",newline="",encoding="utf-8") as fh:
        writer=csv.writer(fh,lineterminator="\n")
        writer.writerow(["question","model","replicate","characters"])
        writer.writerows(sorted(records))
    print(dict(counts))


def analyze():
    from study import DATA,MODELS,digest,evaluate,load_pair,threshold
    p=DATA/"aime_lengths.csv"
    if digest(p)!=LENGTH_SHA:
        raise ValueError("input length digest mismatch")
    with p.open(newline="",encoding="utf-8") as fh:
        values={(r["question"],r["model"],int(r["replicate"])):int(r["characters"]) for r in csv.DictReader(fh)}
    out={}
    for k,s in MODELS.items():
        if s["file"]!="aime.csv":
            continue
        archive,excluded=load_pair(DATA/s["file"],s["models"],s["replicates"])
        for q,(first,_) in archive.items():
            for i,(_,grade) in enumerate(first):
                first[i]=(values[q,s["models"][0],i],grade)
        qs=sorted(archive,key=lambda q:hashlib.sha256(q.encode()).digest())
        pivot=len(qs)//5
        lengths=[n for q in qs[:pivot] for n,_ in archive[q][0]]
        c,m=threshold(lengths,.75),threshold(lengths,.5)
        v=evaluate(qs[pivot:],archive,c,m)
        # These four quantities have first-stage character units in this sensitivity.
        for field in ("first_tokens","capped_first_tokens","random_bypass_first_tokens","streaming_minus_bypass_first_tokens"):
            v.pop(field)
        out[k]=dict(cut_characters=c,median_characters=m,evaluation=v,excluded=excluded)
    dest=Path("reports/aime_characters.json")
    dest.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    print("character sensitivity",digest(dest))


LENGTH_SHA="aa70ad53eab554e789a65defded0ef805b88aa73f0c2588446a9175626c05b5e"
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path)
    args=parser.parse_args()
    if args.project:
        project(args.project)
    else:
        analyze()
