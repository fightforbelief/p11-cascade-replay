"""Sharp spend-matched excess score using a linear assignment support oracle."""
import json
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from replay import assignment
from study import DATA, MODELS, load_pair


def efficient(points):
    """Vertices of the increasing concave direct-service cost/score envelope."""
    hull=[]
    for c,a in sorted(points,key=lambda p:(p[0],-p[1])):
        if hull and (c==hull[-1][0] or a<=hull[-1][1]):
            continue
        while len(hull)>=2:
            x,y=hull[-2]
            u,v=hull[-1]
            if (v-y)*(c-u)>(a-v)*(u-x):
                break
            hull.pop()
        hull.append((c,a))
    return hull


def slopes(hull):
    return [0.]+[(b-a)/(d-c) for (c,a),(d,b) in zip(hull,hull[1:])]


def direct_value(hull,cost):
    if cost < hull[0][0]-1e-8:
        raise ValueError("no direct service affordable at this spend")
    for (c,a),(d,b) in zip(hull,hull[1:]):
        if cost<d:
            return a+(b-a)*(cost-c)/(d-c)
    return hull[-1][1]


def certificate(points,oracle,tolerance=1e-8):
    """oracle(price, minimize) returns attainable (cost,score) for score-price*cost.

    The entire replay set must afford the cheapest direct service. Values use
    points as score units. Return a primal witness and a dual lower bound.
    """
    hull=efficient(points)
    prices=slopes(hull)
    def gap(p):
        return p[1]-direct_value(hull,p[0])
    def intercept(lam):
        return max(y-lam*x for x,y in hull)
    best=max((oracle(lam,False) for lam in prices),key=gap)
    support=lru_cache(maxsize=None)(lambda lam:oracle(lam,True))
    witnesses=set()
    lower=float("-inf")
    dual_price=0.
    def probe(lam):
        nonlocal lower,dual_price
        p=support(lam)
        witnesses.add(p)
        val=p[1]-lam*p[0]-intercept(lam)
        if val>lower:
            lower,dual_price=val,lam
        direct=max(hull,key=lambda v:v[1]-lam*v[0])
        return direct[0]-p[0]
    for lam in prices:
        probe(lam)
    left,right=0.,max(prices)
    for _ in range(54):
        mid=(left+right)/2
        if mid==left or mid==right:
            break
        if probe(mid)>0:
            left=mid
        else:
            right=mid
    upper_point=min(witnesses,key=gap)
    primal=gap(upper_point)
    evidence=[(1.,upper_point)]
    # The minimum on a witness segment occurs at an endpoint or a direct kink.
    for p,q in combinations(sorted(witnesses),2):
        for c,_ in hull:
            if p[0]<c<q[0]:
                w=(c-p[0])/(q[0]-p[0])
                score=p[1]+w*(q[1]-p[1])
                value=score-direct_value(hull,c)
                if value<primal:
                    primal=value
                    evidence=[(1-w,p),(w,q)]
    if not -tolerance <= primal-lower <= tolerance:
        raise ArithmeticError(f"unclosed support certificate: {primal-lower}")
    return dict(minimum_pp=primal,maximum_pp=gap(best),minimum_dual_pp=lower,
                proof_gap_pp=primal-lower,dual_price=dual_price,
                minimum_mixture=evidence,maximum_point=best)


class GateReplay:
    """Cost and grade travel with the same fallback output; first call is always paid."""
    def __init__(self,archive,questions,cutoff,price):
        self.items=[]
        m=len(archive[questions[0]][0])
        self.m=m
        self.scale=m*len(questions)
        self.c0=self.a0=0.
        direct_scores=[0.,0.]
        fallback_total=0.
        self.cost_mat=[[0.]*m for _ in range(m)]
        self.score_mat=[[0.]*m for _ in range(m)]
        self.product=[0.,0.]
        for q in questions:
            first,second=archive[q]
            flags=[t>cutoff for t,y in first]
            k=sum(flags)
            self.c0+=sum(t for t,y in first)
            self.a0+=sum(100*y for flag,(t,y) in zip(flags,first) if not flag)
            direct_scores[0]+=sum(100*y for t,y in first)
            costs=[price*t for t,y in second]
            scores=[100*y for t,y in second]
            direct_scores[1]+=sum(scores)
            fallback_total+=sum(costs)
            self.product[0]+=k*sum(costs)/m
            self.product[1]+=k*sum(scores)/m
            if k:
                self.items.append((k,list(zip(costs,scores))))
            for i,flag in enumerate(flags):
                if flag:
                    for j,(t,y) in enumerate(zip(costs,scores)):
                        self.cost_mat[i][j]+=t
                        self.score_mat[i][j]+=y
        self.direct=[(self.c0/self.scale,direct_scores[0]/self.scale),
                     (fallback_total/self.scale,direct_scores[1]/self.scale)]
        self.product=self.normalize(*self.product)

    def normalize(self,cost,score):
        return (self.c0+cost)/self.scale,(self.a0+score)/self.scale

    def local(self,lam,minimize):
        cost=score=0.
        for k,draws in self.items:
            selected=draws if k==self.m else sorted(draws,key=lambda p:p[1]-lam*p[0],reverse=not minimize)[:k]
            for c,a in selected:
                cost+=c
                score+=a
        return self.normalize(cost,score)

    def common(self,lam,minimize):
        sign=1 if minimize else -1
        matrix=[[sign*(a-lam*c) for c,a in zip(cs,ys)] for cs,ys in zip(self.cost_mat,self.score_mat)]
        perm=assignment(matrix)
        return self.normalize(sum(self.cost_mat[i][j] for i,j in enumerate(perm)),
                              sum(self.score_mat[i][j] for i,j in enumerate(perm)))


def main():
    original=json.loads(Path("reports/results.json").read_text())
    report={}
    for key in ("nano-super","nano-llama","aime-medium"):
        model=MODELS[key]
        data,excluded=load_pair(DATA/model["file"],model["models"],model["replicates"])
        assert not excluded
        test=sorted(set(data)-set(original[key]["calibration"]))
        report[key]={}
        for rho in (1,4):
            replay=GateReplay(data,test,original[key]["cut"],rho)
            hull=efficient(replay.direct)
            x,y=replay.product
            result=dict(direct=replay.direct,product=replay.product,product_gap_pp=y-direct_value(hull,x))
            for name in ("local","common"):
                result[name]=certificate(replay.direct,getattr(replay,name))
            report[key][rho]=result
    Path("reports/spend_replay.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")


if __name__=="__main__":
    main()
