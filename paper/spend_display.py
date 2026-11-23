"""Typeset the equal-spend results from the verified research reports."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent


def report(name):
    return json.loads((HERE.parent / "reports" / (name + ".json")).read_text())


def write_table(filename, columns, alignment, rows):
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule"]
    lines += [" & ".join(columns) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (HERE / filename).write_text("\n".join(lines) + "\n")


def build():
    spend = report("spend_replay")
    table = []
    for case, label, prices in [
        ("aime-medium", "AIME 14B to 32B", ["1", "4"]),
        ("nano-super", "MATH Nano to Super", ["1"]),
        ("nano-llama", "MATH Nano to Llama", ["1"]),
    ]:
        for price in prices:
            r = spend[case][price]
            price_label = "1, 4" if case.startswith("nano") else price
            fields = [label, price_label, f"{r['product_gap_pp']:+.2f}"]
            for kind in ("common", "local"):
                lo, hi = r[kind]["minimum_pp"], r[kind]["maximum_pp"]
                fields.append(f"[{lo:+.2f}, {hi:+.2f}]")
            table.append(fields)
    write_table("spend-table.tex", ["Configuration", r"$\rho$", "Product", "Common", "Per question"],
                "lrrrr", table)
    rows = []
    names = {"AGNEWS": "AGNEWS", "COQA": "CoQA", "HEADLINES": "HEADLINES",
             "OVERRULING": "OVERRULING", "SCIQ": "SciQ"}
    for task, arms in report("published_frontiers").items():
        r = arms["native"]
        unique = r["unique_test_envelope"]
        gaps = [p["hull_gap_pp"] for p in r["rows"]]
        rows.append([names[task], str(unique["comparable"]), str(unique["mixture_wins"]),
                     str(unique["mixture_wins_over_one_point"]),
                     f"[{min(gaps):.2f}, {max(gaps):.2f}]"])
    write_table("external-table.tex", ["Task", "Distinct", "Positive", r"$>1$ point", "Range (points)"],
                "lrrrr", rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    for ax, price in zip(axes, ("1", "4")):
        spec = spend["aime-medium"][price]
        (s, a), (l, b) = spec["direct"]
        c, y = spec["product"]
        xend = max(l, c) * 1.04
        value = a + (b-a) * min(1, (c-s)/(l-s))
        ax.plot([s/1000, l/1000, xend/1000], [a, b, b], color="#2f6784", linewidth=2)
        ax.scatter([s/1000, l/1000], [a, b], color="#2f6784", s=26, zorder=3)
        ax.plot([c/1000, c/1000], [value, y], color="#b75c36", linestyle="--")
        ax.scatter(c/1000, y, marker="D", color="#b75c36", s=43, zorder=4)
        ax.set_title("Fallback / first token price = " + price, fontsize=10)
        ax.set_xlabel("Expected spend (thousands)", fontsize=9)
        ax.set_ylim(48.8, 62.3)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.16)
        ax.annotate(f"{y-value:+.2f} pp", (c/1000, (y+value)/2), xytext=(-7, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8)
    axes[0].set_ylabel("Accuracy (%)", fontsize=9)
    fig.legend(handles=[
        Line2D([0], [0], color="#2f6784", marker="o", label="Best affordable direct mixture"),
        Line2D([0], [0], color="#b75c36", marker="D", linestyle="none", label="Independent gate replay")],
        loc="lower center", frameon=False, ncol=2, fontsize=8)
    fig.subplots_adjust(left=.085, right=.98, top=.89, bottom=.31, wspace=.2)
    fig.savefig(HERE / "spend.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    build()
