"""Compare released FrugalGPT operating points with feasible direct mixtures."""
import csv
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path("data/frugalgpt")
EPS = Fraction(1,10**12)


def direct_options(points, cap):
    """Extreme feasible mixtures (one direct service, or a pair straddling cap)."""
    for model,(cost,score) in sorted(points.items()):
        if cost<=cap:
            yield score, cost, [(model,Fraction(1))]
    for (m,(c,a)),(n,(d,b)) in itertools.combinations(sorted(points.items()),2):
        if min(c,d)<cap<max(c,d):
            w=(cap-c)/(d-c)
            yield a*(1-w)+b*w,cap,[(m,1-w),(n,w)]


def best(options):
    opts=list(options)
    return max(opts,key=lambda o:(o[0],-o[1],-len(o[2]))) if opts else None


def mixture_value(weights,points):
    return tuple(sum(w*points[name][j] for name,w in weights) for j in (0,1))


def audit_row(row,train,test):
    cap=Fraction(row["Budget"])
    cost,score=Fraction(row["Test_cost"]),Fraction(row["Test_acc"])
    options=list(direct_options(test,cost))
    mix=best(options)
    single=best(o for o in options if len(o[2])==1)
    chosen=best(direct_options(train,cap))
    result={"budget":float(cap),"cascade_test_cost":float(cost),"cascade_test_score":float(score),
            "cascade_within_budget":cost<=cap+EPS}
    result["gap_above_best_at_any_spend_pp"]=float(100*(score-max(a for c,a in test.values())))
    if mix:
        result.update(single_gap_pp=float(100*(score-single[0])),hull_gap_pp=float(100*(score-mix[0])),
                      oracle_weights={k:float(w) for k,w in mix[2]})
    if chosen:
        c,a=mixture_value(chosen[2],test)
        result.update(frozen_cost=float(c),frozen_score=float(a),frozen_weights={k:float(w) for k,w in chosen[2]},
                      frozen_within_budget=c<=cap+EPS,
                      frozen_dominates=c<=cost+EPS and a>=score-EPS and (c<cost-EPS or a>score+EPS),
                      frozen_gap_pp=float(100*(score-a)))
    return result


def summarize(rows):
    comparable=[r for r in rows if "hull_gap_pp" in r]
    frozen=[r for r in rows if "frozen_cost" in r]
    erased=[r for r in comparable if r["single_gap_pp"]>1e-8 and r["hull_gap_pp"]<=1e-8]
    return dict(rows=len(rows),unique=len({(r["cascade_test_cost"],r["cascade_test_score"]) for r in rows}),
                comparable=len(comparable),single_wins=sum(r["single_gap_pp"]>1e-8 for r in comparable),
                mixture_wins=sum(r["hull_gap_pp"]>1e-8 for r in comparable),
                mixture_wins_over_one_point=sum(r["hull_gap_pp"]>1+1e-8 for r in comparable),
                erased=len(erased),frozen_available=len(frozen),
                frozen_dominates=sum(r["frozen_dominates"] for r in frozen),
                frozen_dominates_by_one=sum(r["frozen_dominates"] and r["frozen_gap_pp"]< -1-1e-8 for r in frozen),
                both_feasible=sum(r["frozen_within_budget"] and r["cascade_within_budget"] for r in frozen),
                cascade_over_budget=sum(not r["cascade_within_budget"] for r in rows))


def main():
    report={}
    for task,entry in json.loads((ROOT/"manifest.json").read_text()).items():
        path=ROOT/entry["table"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry["sha256"]
        with path.open(encoding="utf-8-sig",newline="") as fh:
            rows=list(csv.DictReader(fh))
        cascades=[r for r in rows if r["Provider"]=="FrugalGPT"]
        direct=[r for r in rows if r["Provider"]!="FrugalGPT"]
        assert len(cascades)==20 and len(direct)==len(set(r["Method"] for r in direct))==10
        assert len(set(r["Test_size"] for r in rows))==1
        report[task]={}
        for pool,names in (("native",entry["pool"]),("catalogue",[r["Method"] for r in direct])):
            def points(split):
                return {r["Method"]:(Fraction(r[split+"_cost"]),Fraction(r[split+"_acc"]))
                        for r in direct if r["Method"] in names}
            train,test=points("Train"),points("Test")
            assert set(train)==set(names)
            checks=[audit_row(r,train,test) for r in cascades]
            unique={(r["cascade_test_cost"],r["cascade_test_score"]):r for r in checks}
            report[task][pool]={"summary":summarize(checks),"unique_test_envelope":{
                k:v for k,v in summarize(list(unique.values())).items() if k in ("comparable","single_wins","mixture_wins","erased","mixture_wins_over_one_point")},
                "rows":checks,"test_n":int(direct[0]["Test_size"])}
    dest=Path("reports/published_frontiers.json")
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    for task,arms in report.items():
        print(task,arms["native"]["summary"])


if __name__=="__main__":
    main()
