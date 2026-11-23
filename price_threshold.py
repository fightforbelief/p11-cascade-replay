"""Price-ratio decision boundaries for a gate whose quality falls between both services."""
import json
from pathlib import Path
from spend_replay import GateReplay
from study import DATA, MODELS, load_pair


def price_boundary(replay, pick):
    (s,small),(large_length,large)=replay.direct
    delta=large-small
    if delta<=0:
        raise ValueError("increasing direct quality required")
    # Cost is s + paid fallback tokens because replay was constructed at unit ratio.
    def margin(price):
        cost,score=pick(price)
        return score-small-price*(cost-s)
    if margin(0)<=0:
        raise ValueError("the gate must improve on the small model under all alignments")
    at_infinity=delta/large_length
    if margin(at_infinity)<=1e-12:
        return None
    left=s/large_length
    right=max(1.,2*left)
    def wins(rho):
        return margin(delta/(large_length-s/rho))>0
    while not wins(right):
        right*=2
    for _ in range(60):
        mid=(left+right)/2
        if mid==left or mid==right:
            break
        if wins(mid):
            right=mid
        else:
            left=mid
    return dict(lower=left,upper=right)


def main():
    model=MODELS["aime-medium"]
    r=json.loads(Path("reports/results.json").read_text())["aime-medium"]
    data,_=load_pair(DATA/model["file"],model["models"],model["replicates"])
    q=sorted(set(data)-set(r["calibration"]))
    replay=GateReplay(data,q,r["cut"],1)
    lo=replay.local(0,True)[1]
    hi=replay.local(0,False)[1]
    assert replay.direct[0][1]<lo<=hi<replay.direct[1][1]
    report={"product":price_boundary(replay,lambda p:replay.product)}
    for name in ("common","local"):
        oracle=getattr(replay,name)
        report[name]={label:price_boundary(replay,lambda p:oracle(p,minimize))
                      for label,minimize in [("some",False),("all",True)]}
    Path("reports/price_threshold.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(report)


if __name__=="__main__":
    main()
