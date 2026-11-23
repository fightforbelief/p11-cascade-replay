"""Independently check returned result identities, data hashes, and endpoint ordering."""
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    print(f"Verified {count} contrast/direction records and source digests.")


if __name__=="__main__":
    main()
