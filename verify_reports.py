"""Independently check returned result identities, data hashes, and endpoint ordering."""
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crosscheck_accuracy(entry, cutoff, spec, calibration, length_map=None):
    # Direct binary-overlap counts, independent of sorting and assignment code.
    rows=defaultdict(dict)
    with (Path("data")/spec["file"]).open(newline="",encoding="utf-8") as inp:
        for row in csv.DictReader(inp):
            q,model=row["question"],row["model"]
            if model in spec["models"] and q not in calibration:
                i=int(row["replicate"])
                t=int(row["tokens"]) if length_map is None else length_map[q,model,i]
                rows[q,model][i]=(int(row["correct"]),t)
    m=spec["replicates"]
    n=entry["n"]
    questions=sorted({q for q,model in rows})
    assert len(questions)==n
    accepted=0
    lower=upper=mean=identity=sum_first=sum_fallback=penalty=0
    for q in questions:
        first,fallback=(rows[q,model] for model in spec["models"])
        flags=[int(first[i][1]>cutoff) for i in range(m)]
        upgrades=sum(flags)
        z=sum(fallback[i][0] for i in range(m))
        accepted+=upgrades
        sum_first+=sum(first[i][0] for i in range(m))
        sum_fallback+=z
        penalty+=sum(flags[i]*first[i][0] for i in range(m))
        lower+=max(0,upgrades+z-m)
        upper+=min(upgrades,z)
        mean+=upgrades*z/m
        identity+=sum(flags[i]*fallback[i][0] for i in range(m))
    rate=accepted/(n*m)
    assert abs(rate-entry["call_fraction"])<1e-10
    reference=rate*(sum_fallback-sum_first)+penalty
    out=entry["comparisons"]["random"]["accuracy_pp"]
    for key,total in (("product",mean),("identity",identity),("local_min",lower),("local_max",upper)):
        assert abs(out[key]-(total-reference)*100/(m*n))<1e-8,(key,out[key])


def main():
    run=json.loads(Path("reports/execution.json").read_text())
    resultpath=Path("reports/results.json")
    assert sha(resultpath)==run["result_sha256"]
    assert all(sha(p)==v for p,v in run["source"].items())
    data=json.loads(resultpath.read_text())
    directions=("accuracy_pp","tokens","utility_0.1","utility_1","utility_10")
    count=0
    for entry in data.values():
        spec=entry["spec"]
        assert sha(Path("data")/spec["file"])==spec["sha256"]
        for split in ("evaluation","full"):
            v=entry[split]
            for p,metrics in v["comparisons"].items():
                assert set(metrics)==set(directions)
                for name,s in metrics.items():
                    for key in ("product","identity"):
                        chain=[s["local_min"],s["common_min"],s[key],s["common_max"],s["local_max"]]
                        assert all(a<=b+1e-7 for a,b in zip(chain,chain[1:])),(split,p,name,chain)
                    assert s["common_sd"]>=0 and s["local_sd"]>=0
                    for w in ("min","max"):
                        assert sorted(s[f"common_{w}_permutation"])==list(range(spec["replicates"]))
                    if s["global_permutation_signs"] is not None:
                        assert sum(s["global_permutation_signs"].values())==math.factorial(spec["replicates"])
                    count+=1
                for loss in (.1,1,10):
                    for key in ("product","identity"):
                        expected=metrics["accuracy_pp"][key]-loss*metrics["tokens"][key]/1000
                        assert abs(expected-metrics[f"utility_{loss}"][key])<1e-7
                    for alignment in ("common","local"):
                        rect=metrics["accuracy_pp"][alignment+"_min"]-loss*metrics["tokens"][alignment+"_max"]/1000
                        assert rect<=metrics[f"utility_{loss}"][alignment+"_min"]+1e-7
            assert abs(v["streaming_minus_bypass_first_tokens"]-v["capped_first_tokens"]+v["random_bypass_first_tokens"])<1e-7
    with Path("data/aime_lengths.csv").open(encoding="utf-8",newline="") as fh:
        lengths={(r["question"],r["model"],int(r["replicate"])):int(r["characters"]) for r in csv.DictReader(fh)}
    sensitivity=json.loads(Path("reports/aime_characters.json").read_text())
    for name,r in data.items():
        crosscheck_accuracy(r["evaluation"],r["cut"],r["spec"],set(r["calibration"]))
        if name in sensitivity:
            alt=sensitivity[name]
            crosscheck_accuracy(alt["evaluation"],alt["cut_characters"],r["spec"],set(r["calibration"]),lengths)
    print("Independent binary-overlap checks passed for all six evaluation configurations.")
    print(f"Verified {count} contrast/direction records and source digests.")


if __name__=="__main__":
    main()
