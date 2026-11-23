"""Replay the historical AutoMix slope metric and equal-budget family comparisons."""
import hashlib
import itertools
import json
from pathlib import Path
from statistics import fmean,pstdev
from spend_replay import direct_value,efficient

SOURCE=Path("data/automix/paper_eval_multi_seed_final.json")
DIGEST="845374d06c3c0d573320b2c76d2b269e8bd40f26d56d3d87e0e7a39698c86441"


def family(key):
    if "AutomixUnion" in key:
        return "POMDP"
    if "SelfConsistency" in key:
        return "SelfConsistency"
    if "Threshold" in key:
        return "Threshold"
    raise ValueError(key)


def recover(rows):
    # mean[(A-S)/(C-1)] = (1+mean lift)*direct slope.
    equations=[]
    for rec in rows.values():
        pairs=rec["points"]
        equations.append((fmean(1/(c-1) for c,a in pairs),1+rec["avg_lift"],
                          fmean(a/(c-1) for c,a in pairs)))
    def determinant(pair):
        (x,y,_),(u,v,_)=pair
        return x*v-y*u
    pair=max(itertools.combinations(equations,2),key=lambda pair:abs(determinant(pair)))
    (x,y,z),(u,v,w)=pair
    det=determinant(pair)
    if abs(det)<1e-10:
        raise ValueError("direct endpoints unidentifiable")
    small=(z*v-y*w)/det
    slope=(x*w-z*u)/det
    large=small+49*slope
    if not 0<=small<large<=1:
        raise ValueError("invalid direct-score reconstruction")
    err=0
    for rec in rows.values():
        lifts=[(a-small)/((c-1)*slope)-1 for c,a in rec["points"]]
        for statistic,fn in [("avg_lift",fmean),("min_lift",min),("max_lift",max),("std_lift",pstdev)]:
            err=max(err,abs(rec[statistic]-fn(lifts)))
        for c,a in rec["points"]:
            if abs(c-52)<1e-10:
                err=max(err,abs(a-large))
    if err>1e-7:
        raise ValueError(f"published lift discrepancy {err}")
    return small,large,err


def profile(points,small,large,ranking):
    winner=max(ranking,key=ranking.get)
    curves={k:efficient([(1.,small),(50.,large),p]) for k,p in points.items()}
    grid={1.,50.}
    for curve in curves.values():
        grid.update(c for c,a in curve if 1<c<50)
    def values(c):
        return {k:100*direct_value(h,c) for k,h in curves.items()}
    def loss(c):
        v=values(c)
        return max(v.values())-v[winner]
    xs=sorted(grid)
    for l,r in zip(xs,xs[1:]):
        a,b=values(l),values(r)
        for i,j in itertools.combinations(points,2):
            y,z=a[i]-a[j],b[i]-b[j]
            if y*z<0:
                grid.add(l+(r-l)*y/(y-z))
    xs=sorted(grid)
    integral=0.
    higher=0.
    for l,r in zip(xs,xs[1:]):
        integral+=(r-l)*(loss(l)+loss(r))/2
        if loss((l+r)/2)>1e-9:
            higher+=r-l
    at=max(xs,key=loss)
    fixed={}
    for share in (.25,.5,.75):
        v=values(1+49*share)
        fixed[str(share)]=dict(scores_pp=v,best=[k for k,a in v.items() if max(v.values())-a<1e-8],
                               loss_pp=loss(1+49*share))
    return dict(ibc_winner=winner,mean_loss_pp=integral/49,maximum_loss_pp=loss(at),budget_at_max=at,
                fraction_strictly_suboptimal=higher/49,fixed=fixed)


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==DIGEST
    source=json.loads(SOURCE.read_text())
    report={}
    for task,published in source["avg_lift"].items():
        rows={}
        for raw,avg in published.items():
            method=family(raw)
            assert method not in rows
            pts=list(zip(source["costs"][task][raw],source["perfs"][task][raw],strict=True))
            assert len(pts)==10 and all(2<=c<=52+1e-9 and 0<=a<=1 for c,a in pts)
            rows[method]={k:source[k][task][raw] for k in ("avg_lift","min_lift","max_lift","std_lift")}
            rows[method]["points"]=pts
        assert len(rows)==3
        small,large,err=recover(rows)
        points={k:tuple(map(fmean,zip(*r["points"]))) for k,r in rows.items()}
        ranking={k:r["avg_lift"] for k,r in rows.items()}
        slopes={k:(a-small)/(c-1) for k,(c,a) in points.items()}
        by_seed=[]
        for i in range(10):
            pts={k:r["points"][i] for k,r in rows.items()}
            ra={k:(a-small)/(c-1) for k,(c,a) in pts.items()}
            by_seed.append(profile(pts,small,large,ra))
        report[task]=dict(small=small,large=large,max_reconstruction_error=err,
                          avg_points=points,mean_ibc=ranking,mean_point_slope_winner=max(slopes,key=slopes.get),
                          profile=profile(points,small,large,ranking),by_seed=by_seed)
    Path("reports/automix_frontiers.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    for task,r in report.items():
        p=r["profile"]
        print(task,p["ibc_winner"],p["maximum_loss_pp"],p["fraction_strictly_suboptimal"])


if __name__=="__main__":
    main()
