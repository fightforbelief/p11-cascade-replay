"""Independent checks of tabulated results and a reproduction receipt."""
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from published_frontiers import ROOT
from spend_replay import efficient,direct_value


def load(name):
    return json.loads((Path("reports")/(name+".json")).read_text())


def verify():
    paper=load("published_frontiers")
    metadata=json.loads((ROOT/"manifest.json").read_text())
    for task,arms in paper.items():
        with (ROOT/metadata[task]["table"]).open(newline="",encoding="utf-8-sig") as f:
            records=list(csv.DictReader(f))
        for pool,arm in arms.items():
            pts={r['Method']:(float(r['Test_cost']),float(r['Test_acc'])) for r in records
                 if r['Provider']!='FrugalGPT' and (pool=='catalogue' or r['Method'] in metadata[task]['pool'])}
            hull=efficient(pts.values())
            assert len(arm['rows'])==20
            for r in arm['rows']:
                c,a=r['cascade_test_cost'],r['cascade_test_score']
                assert abs(100*(a-direct_value(hull,c))-r['hull_gap_pp'])<1e-8
                assert abs(100*(a-max(y for x,y in pts.values()))-r['gap_above_best_at_any_spend_pp'])<1e-8
                weights=r.get('frozen_weights')
                if weights:
                    assert abs(sum(weights.values())-1)<1e-12
                    for i,k in enumerate(['frozen_cost','frozen_score']):
                        assert abs(sum(w*pts[m][i] for m,w in weights.items())-r[k])<1e-9
    for row in load('automix_frontiers').values():
        assert row['max_reconstruction_error']<1e-7
        p=row['profile']
        peak=avg=0.
        direct=[(1,row['small']),(50,row['large'])]
        curves={k:efficient(direct+[v]) for k,v in row['avg_points'].items()}
        for i in range(2001):
            x=1+49*i/2000
            y={k:direct_value(curve,x) for k,curve in curves.items()}
            loss=100*(max(y.values())-y[p['ibc_winner']])
            avg+=loss*(.5 if i in (0,2000) else 1)/2000
            peak=max(peak,loss)
        assert abs(avg-p['mean_loss_pp'])<.001 and 0<=p['maximum_loss_pp']-peak+1e-8<.01
    replay=load('spend_replay')
    for arms in replay.values():
        for row in arms.values():
            c,a=row['product']
            assert abs(a-direct_value(efficient(row['direct']),c)-row['product_gap_pp'])<1e-8
            ends=[row['local']['minimum_pp'],row['common']['minimum_pp'],row['product_gap_pp'],
                  row['common']['maximum_pp'],row['local']['maximum_pp']]
            assert all(x<=y+1e-8 for x,y in zip(ends,ends[1:]))
            for kind in ('common','local'):
                r=row[kind]
                assert abs(r['minimum_pp']-r['minimum_dual_pp'])<1e-8
    thresholds=load('price_threshold')
    sample=replay['aime-medium']['1']
    (s,y),(l,z)=sample['direct']
    c,a=sample['product']
    cross=s*(a-y)/(l*(a-y)-(z-y)*(c-s))
    assert abs(cross-thresholds['product']['upper'])<1e-8
    def upper(x):
        return x['upper'] if x else float('inf')
    ordered=[upper(thresholds['local']['some']),upper(thresholds['common']['some']),cross,
             upper(thresholds['common']['all']),upper(thresholds['local']['all'])]
    assert all(x<=y+1e-8 for x,y in zip(ordered,ordered[1:]))
    assert len(load('question_sensitivity'))==6
    print('Verified external envelopes, held-out mixtures, IBC reconstructions, grid integrals and price crossings.')


if __name__=="__main__":
    verify()
    scripts=("spend_replay.py","replay.py","published_frontiers.py","automix_frontiers.py","price_threshold.py",
             "verify_frontiers.py","test_spend.py","FRONTIER_PROTOCOL.md","question_sensitivity.py","study.py","Run-Study.ps1")
    reports=("published_frontiers","automix_frontiers","price_threshold","spend_replay","question_sensitivity")
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    record=dict(python=sys.version,platform=platform.platform(),sources={f:digest(f) for f in scripts},
                results={f:digest(Path("reports")/(f+".json")) for f in reports})
    Path("reports/frontier_execution.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8",newline="\n")
