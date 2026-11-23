"""Build manuscript tables and the range figure from returned Windows results."""
import json
from spend_display import build as build_spend_displays
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
results=json.loads((HERE.parent/"reports/results.json").read_text())
LABELS={"nano-super":"Nano to Super","nano-llama":"Nano to Llama","aime-small":"1.5B to 32B","aime-medium":"14B to 32B"}
rows=list(results.items())


def tabular(file, header, body, align):
    lines=[r"\begin{tabular}{"+align+"}",r"\toprule"," & ".join(header)+r" \\",r"\midrule"]
    lines.extend(" & ".join(r)+r" \\" for r in body)
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (HERE/file).write_text("\n".join(lines)+"\n")


def span(s, prefix, digits=2):
    return f"[{s[prefix+'_min']:.{digits}f}, {s[prefix+'_max']:.{digits}f}]"


tabular("sample.tex",["Configuration","Cal./eval.","Draws","Cutoff","Calls","First / fallback"],
        [[LABELS[k],f"{len(r['calibration'])}/{r['evaluation']['n']}",str(r['spec']['replicates']),
          f"{r['cut']:,}",f"{100*r['evaluation']['call_fraction']:.1f}\\%",
          f"{100*r['evaluation']['first_accuracy']:.1f} / {100*r['evaluation']['fallback_accuracy']:.1f}"] for k,r in rows],"lrrrrr")

def accuracy_row(k,r):
    s=r['evaluation']['comparisons']['random']['accuracy_pp']
    erase=s['erase_questions']
    return [LABELS[k],f"{s['identity']:.2f}",f"{s['product']:.2f}",span(s,'common'),span(s,'local'),
            f"{s['common_sd']:.2f}/{s['local_sd']:.2f}",str(erase) if erase is not None else r"$\infty$"]
tabular("accuracy.tex",["Configuration","ID","Product","Common","Per question","SDs","Erase"],
        [accuracy_row(k,r) for k,r in rows],"lrrrrrr")

def costs_row(k,r):
    v=r["evaluation"]; s=v["comparisons"]["random"]["tokens"];f=v["streaming_minus_bypass_first_tokens"]
    return [LABELS[k],f"{f:.0f}",f"{s['product']:.0f}",span(s,'local',0),
            f"[{f+s['local_min']:.0f}, {f+s['local_max']:.0f}]",
            f"[{f+4*s['local_min']:.0f}, {f+4*s['local_max']:.0f}]"]
tabular("costs.tex",["Configuration",r"$\Delta$ first",r"$\Delta$ fallback","Fallback range",r"$\rho=1$",r"$\rho=4$"],
        [costs_row(k,r) for k,r in rows if k!="aime-small"],"lrrrrr")
tabular("controls.tex",["Configuration","Keep","Median","Always","Within","Between","All questions"],
        [[LABELS[k]]+[f"{r['evaluation']['comparisons'][p]['accuracy_pp']['product']:.2f}" for p in ['keep','median','always','within_question']]+
         [f"{r['evaluation']['comparisons']['random']['accuracy_pp']['product']-r['evaluation']['comparisons']['within_question']['accuracy_pp']['product']:.2f}",
          f"{r['full']['comparisons']['random']['accuracy_pp']['product']:.2f}"] for k,r in rows],"lrrrrrr")
fig,axes=plt.subplots(1,2,figsize=(8.1,2.6),layout="constrained")
for ax,(key,title) in zip(axes,[("nano-super","MATH-500 · Nano to Super"),("aime-medium","AIME · 14B to 32B")]):
    contrast=results[key]["evaluation"]["comparisons"]["random"]
    stats=[("utility_1","Utility at 1"),("utility_0.1","Utility at 0.1"),("accuracy_pp","Accuracy")]
    for y,(metric,label) in enumerate(stats):
        s=contrast[metric]
        ax.plot([s["local_min"],s["local_max"]],[y,y],lw=1.7,c="#52575f")
        ax.plot([s["common_min"],s["common_max"]],[y,y],lw=6,c="#5289ad",solid_capstyle="butt")
        ax.scatter(s["product"],y,s=28,marker="D",c="#92341e",zorder=3)
    ax.axvline(0,c="#454545",ls=":",lw=1,zorder=0)
    ax.set_title(title,fontsize=10);ax.set_yticks(range(3),[label for _,label in stats])
    ax.set_xlabel("Difference (accuracy-point equivalents)",fontsize=8)
    ax.tick_params(labelsize=8);ax.set_ylim(-.55,2.55)
    ax.spines[["top","right","left"]].set_visible(False);ax.grid(axis="x",alpha=.16)
fig.savefig(HERE/"alignment.pdf")

build_spend_displays()
